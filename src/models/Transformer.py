import os
import time
import math
import torch
import torch.nn as nn

from torch.nn import TransformerEncoderLayer
from models.FusionModules import TransformerFusionBlock


class PositionalEncoding(nn.Module):
    def __init__(self, out_channel, dropout=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_channel, 2) * (-math.log(10000.0) / out_channel))
        pe = torch.zeros(max_len, 1, out_channel)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["Transformer"]
        self.device = args.device
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_segments = args.dataset_config["num_segments"]

        # Single mod,  [b, i, s*c]
        self.loc_mod_feature_extraction_layers = nn.ModuleDict()
        for loc in self.locations:
            self.loc_mod_feature_extraction_layers[loc] = nn.ModuleDict()
            for mod in self.modalities:
                spectrum_len = args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                feature_channels = args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                module_list = [nn.Linear(spectrum_len * feature_channels, self.config["loc_mod_out_channels"])] + [
                    TransformerEncoderLayer(
                        d_model=self.config["loc_mod_out_channels"],
                        nhead=self.config["loc_mod_head_num"],
                        dim_feedforward=self.config["loc_mod_out_channels"],
                        dropout=self.config["dropout_ratio"],
                        batch_first=True,
                    )
                    for _ in range(self.config["loc_mod_block_num"])
                ]
                self.loc_mod_feature_extraction_layers[loc][mod] = nn.Sequential(*module_list)

        # Single loc, [b, i, c]
        self.mod_fusion_layers = nn.ModuleDict()
        self.loc_feature_extraction_layers = nn.ModuleDict()
        for loc in self.locations:
            self.mod_fusion_layers[loc] = TransformerFusionBlock(
                self.config["loc_mod_out_channels"],
                self.config["loc_head_num"],
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )
            module_list = [nn.Linear(self.config["loc_mod_out_channels"], self.config["loc_out_channels"])] + [
                TransformerEncoderLayer(
                    d_model=self.config["loc_out_channels"],
                    nhead=self.config["loc_head_num"],
                    dim_feedforward=self.config["loc_out_channels"],
                    dropout=self.config["dropout_ratio"],
                    batch_first=True,
                )
                for _ in range(self.config["loc_block_num"])
            ]
            self.loc_feature_extraction_layers[loc] = nn.Sequential(*module_list)

        # Single interval, [b, i, c]
        self.loc_fusion_layer = TransformerFusionBlock(
            self.config["loc_out_channels"],
            self.config["sample_head_num"],
            self.config["dropout_ratio"],
            self.config["dropout_ratio"],
        )
        module_list = [nn.Linear(self.config["loc_out_channels"], self.config["sample_out_channels"])] + [
            TransformerEncoderLayer(
                d_model=self.config["sample_out_channels"],
                nhead=self.config["sample_head_num"],
                dim_feedforward=self.config["sample_out_channels"],
                dropout=self.config["dropout_ratio"],
                batch_first=True,
            )
            for _ in range(self.config["sample_block_num"])
        ]
        self.sample_feature_extraction_layer = nn.Sequential(*module_list)

        # Time fusion, [b, c]
        self.time_fusion_layer = TransformerFusionBlock(
            self.config["sample_out_channels"],
            self.config["sample_head_num"],
            self.config["dropout_ratio"],
            self.config["dropout_ratio"],
        )

        # Classification
        self.sample_embd_layer = nn.Sequential(
            nn.Linear(self.config["sample_out_channels"], self.config["fc_dim"]),
            nn.GELU(),
        )
        self.class_layer = nn.Sequential(
            nn.Linear(self.config["fc_dim"], args.dataset_config["num_classes"]),
            # nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
        )

    def forward(self, freq_x, class_head=True):
        """The forward function of DeepSense.
        Args:
            time_x (_type_): time_x is a dictionary consisting of the Tensor input of each input modality.
                        For each modality, the data is in (b, c (2 * 3 or 1), i (intervals), s (spectrum)) format.
        """
        args = self.args

        # Step 1: Single (loc, mod) feature extraction, [b, i, s, c]
        loc_mod_features = dict()
        for loc in self.locations:
            loc_mod_features[loc] = []
            for mod in self.modalities:
                # [b, c, i, s] -- > [b, i, s, c]
                loc_mod_input = torch.permute(freq_x[loc][mod], [0, 2, 3, 1])
                b, i, s, c = loc_mod_input.shape
                loc_mod_input = torch.reshape(loc_mod_input, (b, i, s * c))
                loc_mod_features[loc].append(self.loc_mod_feature_extraction_layers[loc][mod](loc_mod_input))
            loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=2)

        # Step 2: Modality-level fusion
        loc_fused_features = {}
        for loc in loc_mod_features:
            loc_fused_features[loc] = self.mod_fusion_layers[loc](loc_mod_features[loc])

        # Step 3: Location feature extraction, [b, i, s, c]
        loc_features = []
        for loc in loc_mod_features:
            outputs = self.loc_feature_extraction_layers[loc](loc_fused_features[loc])
            loc_features.append(outputs)
        loc_features = torch.stack(loc_features, dim=2)

        # Step 4: Location-level fusion, [b, i, c]
        interval_features = self.loc_fusion_layer(loc_features)
        interval_features = self.sample_feature_extraction_layer(interval_features)
        interval_features = torch.unsqueeze(interval_features, dim=1)

        # Step 5: Time fusion
        sample_features = self.time_fusion_layer(interval_features)
        sample_features = torch.flatten(sample_features, start_dim=1)

        # Step 6: Classification
        outputs = torch.flatten(sample_features, start_dim=1)
        sample_features = self.sample_embd_layer(outputs)

        if class_head:
            logits = self.class_layer(sample_features)
            return logits
        else:
            """Self-supervised pre-training"""
            return sample_features

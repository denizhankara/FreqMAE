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
        self.out_channel = out_channel

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
        x = x.permute(1, 0, 2) * math.sqrt(self.out_channel)
        x = x + self.pe[: x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class TransformerV3(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["TransformerV2"]
        self.device = args.device
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_segments = args.dataset_config["num_segments"]

        # decide max seq len
        max_len = 0
        for loc in self.locations:
            for mod in self.modalities:
                stride = self.config["in_stride"][mod]
                max_len = max(max_len, int(args.dataset_config["loc_mod_spectrum_len"][loc][mod] / stride))

        # Single mod,  [b * i, s/stride, c * stride]
        self.freq_context_layers = nn.ModuleDict()
        self.freq_fusion_layers = nn.ModuleDict()
        self.interval_context_layers = nn.ModuleDict()
        self.interval_fusion_layers = nn.ModuleDict()

        for loc in self.locations:
            self.freq_context_layers[loc] = nn.ModuleDict()
            self.freq_fusion_layers[loc] = nn.ModuleDict()
            self.interval_context_layers[loc] = nn.ModuleDict()
            self.interval_fusion_layers[loc] = nn.ModuleDict()

            for mod in self.modalities:
                stride = self.config["in_stride"][mod]
                input_channels = args.dataset_config["loc_mod_in_freq_channels"][loc][mod]

                """Single (loc, mod, freq) feature extraction layer"""
                module_list = [
                    nn.Linear(int(stride * input_channels), self.config["freq_out_channels"]),
                    # PositionalEncoding(self.config["freq_out_channels"], 0.1, max_len),
                ] + [
                    TransformerEncoderLayer(
                        d_model=self.config["freq_out_channels"],
                        nhead=self.config["freq_head_num"],
                        dim_feedforward=self.config["freq_out_channels"],
                        dropout=self.config["dropout_ratio"],
                        batch_first=True,
                    )
                    for _ in range(self.config["freq_block_num"])
                ]
                self.freq_context_layers[loc][mod] = nn.Sequential(*module_list)

                """Freq fusion layer for each (loc, mod)"""
                self.freq_fusion_layers[loc][mod] = TransformerFusionBlock(
                    self.config["freq_out_channels"],
                    self.config["freq_head_num"],
                    self.config["dropout_ratio"],
                    self.config["dropout_ratio"],
                )

                module_list = [
                    nn.Linear(self.config["freq_out_channels"], self.config["interval_out_channels"]),
                    # PositionalEncoding(self.config["interval_out_channels"], 0.1, max_len),
                ] + [
                    TransformerEncoderLayer(
                        d_model=self.config["interval_out_channels"],
                        nhead=self.config["interval_head_num"],
                        dim_feedforward=self.config["interval_out_channels"],
                        dropout=self.config["dropout_ratio"],
                        batch_first=True,
                    )
                    for _ in range(self.config["interval_block_num"])
                ]
                self.interval_context_layers[loc][mod] = nn.Sequential(*module_list)
                self.interval_fusion_layers[loc][mod] = TransformerFusionBlock(
                    self.config["interval_out_channels"],
                    self.config["interval_head_num"],
                    self.config["dropout_ratio"],
                    self.config["dropout_ratio"],
                )

        # Single loc, [b, i, c]
        self.mod_context_layers = nn.ModuleDict()
        self.mod_fusion_layer = nn.ModuleDict()
        for loc in self.locations:
            """Single (loc, mod) feature extraction"""
            module_list = [nn.Linear(self.config["interval_out_channels"], self.config["loc_mod_out_channels"])] + [
                TransformerEncoderLayer(
                    d_model=self.config["loc_mod_out_channels"],
                    nhead=self.config["loc_mod_head_num"],
                    dim_feedforward=self.config["loc_mod_out_channels"],
                    dropout=self.config["dropout_ratio"],
                    batch_first=True,
                )
                for _ in range(self.config["loc_mod_block_num"])
            ]
            self.mod_context_layers[loc] = nn.Sequential(*module_list)

            """Mod fusion layer for each loc"""
            self.mod_fusion_layer[loc] = TransformerFusionBlock(
                self.config["loc_mod_out_channels"],
                self.config["loc_mod_head_num"],
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )

        # Single sample
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
        self.loc_context_layer = nn.Sequential(*module_list)
        self.loc_fusion_layer = TransformerFusionBlock(
            self.config["loc_out_channels"],
            self.config["loc_head_num"],
            self.config["dropout_ratio"],
            self.config["dropout_ratio"],
        )

        # Classification
        self.sample_emdb_layer = nn.Sequential(
            nn.Linear(self.config["loc_out_channels"], self.config["fc_dim"]),
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

        # Step 1: Single (loc, mod, freq) feature extraction, [b * i, int(s / stride), stride * c]
        loc_mod_features = dict()
        for loc in self.locations:
            loc_mod_features[loc] = []
            for mod in self.modalities:
                """freq feature extraction"""
                # [b, c, i, spectrum] -- > [b, i, spectrum, c]
                freq_input = torch.permute(freq_x[loc][mod], [0, 2, 3, 1])
                b, i, s, c = freq_input.shape
                stride = self.config["in_stride"][mod]
                freq_input = torch.reshape(freq_input, (b * i, -1, stride * c))

                freq_context_feature = self.freq_context_layers[loc][mod](freq_input)
                freq_context_feature = freq_context_feature.reshape(
                    [b, i, int(s / stride), self.config["freq_out_channels"]]
                )

                """freq feature fusion, [b, i, c]"""
                interval_input = self.freq_fusion_layers[loc][mod](freq_context_feature)

                """interval feature extraction, [b, 1, i, c]"""
                interval_context_feature = self.interval_context_layers[loc][mod](interval_input)
                interval_context_feature = interval_context_feature.unsqueeze(1)
                loc_mod_input = self.interval_fusion_layers[loc][mod](interval_context_feature)
                loc_mod_features[loc].append(loc_mod_input)

            # [b, 1, sensor, c]
            loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=2)

        # Step 2: Loc mod feature extraction, [b, i, location, c]
        loc_features = []
        for loc in loc_mod_features:
            """Extract mod feature with peer-feature context"""
            b, i, mods, c = loc_mod_features[loc].shape
            loc_mod_input = loc_mod_features[loc].reshape([b * i, mods, c])
            loc_mod_context_feature = self.mod_context_layers[loc](loc_mod_input)
            loc_mod_context_feature = loc_mod_context_feature.reshape([b, i, mods, c])

            """Mod feature fusion, [b, 1, 1, c]"""
            loc_feature = self.mod_fusion_layer[loc](loc_mod_context_feature)
            loc_features.append(loc_feature)
        loc_features = torch.stack(loc_features, dim=2)

        # Step 3: Location-level fusion, [b, 1, l, c]
        if len(self.locations) > 1:
            b, i, l, c = loc_features.shape
            loc_input = loc_features.reshape([b * i, l, c])
            loc_context_feature = self.loc_context_layer(loc_input)
            loc_context_feature = loc_context_feature.reshape([b, i, l, c])
            sample_features = self.loc_fusion_layer(loc_context_feature)
        else:
            sample_features = loc_features.squeeze(dim=2)

        # Step 4: Classification
        sample_features = self.sample_emdb_layer(sample_features)
        if class_head:
            logits = self.class_layer(sample_features)
            return logits
        else:
            """Self-supervised pre-training"""
            return sample_features

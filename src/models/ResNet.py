import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from models.ConvModules import ConvBlock
from models.FusionModules import MeanFusionBlock


class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = args.dataset_config["ResNet"]
        self.device = args.device
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.multi_location_flag = len(self.locations) > 1

        """(loc, mod) feature extraction"""
        self.loc_mod_extractors = nn.ModuleDict()
        for loc in self.locations:
            self.loc_mod_extractors[loc] = nn.ModuleDict()
            for mod in self.modalities:
                if type(self.config["loc_mod_conv_lens"]) is dict:
                    """for acoustic processing in Parkland data"""
                    conv_lens = self.config["loc_mod_conv_lens"][mod]
                    in_stride = self.config["loc_mod_in_conv_stride"][mod]
                else:
                    conv_lens = self.config["loc_mod_conv_lens"]
                    in_stride = 1

                # define the extractor
                self.loc_mod_extractors[loc][mod] = ConvBlock(
                    in_channels=args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    out_channels=self.config["loc_mod_out_channels"],
                    in_spectrum_len=args.dataset_config["loc_mod_spectrum_len"][loc][mod],
                    interval_num=self.config["interval_num"],
                    conv_lens=conv_lens,
                    dropout_ratio=self.config["dropout_ratio"],
                    num_inter_layers=self.config["loc_mod_conv_inter_layers"],
                    in_stride=in_stride,
                )

        """mod feature fusion and loc feature extraction"""
        self.mod_fusion_layers = nn.ModuleDict()
        for loc in self.locations:
            self.mod_fusion_layers[loc] = MeanFusionBlock()
        self.loc_extractors = nn.ModuleDict()
        for loc in self.locations:
            self.loc_extractors[loc] = ConvBlock(
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_mod_out_channels"],
                interval_num=1,
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )

        """loc feature fusion and sample feature extraction"""
        if self.multi_location_flag:
            self.loc_fusion_layer = MeanFusionBlock()
            self.interval_extractor = ConvBlock(
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_out_channels"],
                interval_num=1,
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )

        # Step 5: Classification layer
        self.sample_embd_layer = nn.Sequential(
            nn.Linear(self.config["loc_out_channels"], self.config["fc_dim"]),
            nn.ReLU(),
        )
        self.class_layer = nn.Sequential(
            nn.Linear(self.config["fc_dim"], args.dataset_config["num_classes"]),
            # nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
        )

    def forward(self, freq_x, class_head=True):
        """The forward function of ResNet.
        Args:
            x (_type_): x is a dictionary consisting of the Tensor input of each input modality.
                        For each modality, the data is in (b, c (2 * 3 or 1), i (intervals), s (spectrum)) format.
        """
        # Step 4: Single (loc, mod) feature extraction
        loc_mod_features = dict()
        for loc in self.locations:
            loc_mod_features[loc] = []
            for mod in self.modalities:
                loc_mod_features[loc].append(self.loc_mod_extractors[loc][mod](freq_x[loc][mod]))
            loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=3)

        # Step 4: Feature fusion for different mods in the same location
        fused_loc_features = dict()
        for loc in self.locations:
            fused_loc_features[loc] = self.mod_fusion_layers[loc](loc_mod_features[loc])

        # Step 5: Feature extraction for each location
        extracted_loc_features = dict()
        for loc in self.locations:
            extracted_loc_features[loc] = self.loc_extractors[loc](fused_loc_features[loc])

        # Step 6: Location fusion, (b, c, i)
        if not self.multi_location_flag:
            final_feature = extracted_loc_features[self.locations[0]]
        else:
            interval_fusion_input = torch.stack([extracted_loc_features[loc] for loc in self.locations], dim=3)
            fused_feature = self.loc_fusion_layer(interval_fusion_input)
            final_feature = self.interval_extractor(fused_feature)

        # Step 7: Classification on features
        final_feature = torch.flatten(final_feature, start_dim=1)
        sample_features = self.sample_embd_layer(final_feature)

        if class_head:
            logits = self.class_layer(sample_features)
            return logits
        else:
            """Self-supervised pre-training"""
            return sample_features

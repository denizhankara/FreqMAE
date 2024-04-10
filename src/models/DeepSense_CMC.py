import os
import time
import torch
import torch.nn as nn

from models.ConvModules import ConvBlock, DeConvBlock, FIC
from models.FusionModules import MeanFusionBlock, SelfAttentionFusionBlock
from models.RecurrentModule import RecurrentBlock, DecRecurrentBlock
from input_utils.normalize import normalize_input
from input_utils.padding_utils import get_padded_size
from input_utils.mask_utils import mask_input
from models.FusionModules import TransformerFusionBlock
from models.GMCModules import Swish


class DeepSense_CMC(nn.Module):
    def __init__(self, args, self_attention=False) -> None:
        """The initialization for the DeepSense class.
        Design: Single (interval, loc, mod) feature -->
                Single (interval, loc) feature -->
                Single interval feature -->
                GRU -->
                Logits
        Args:
            num_classes (_type_): _description_
        """
        super().__init__()
        self.args = args
        self.self_attention = self_attention
        self.config = args.dataset_config["DeepSense"]
        self.device = args.device
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.multi_location_flag = len(self.locations) > 1

        """define the architecture"""
        self.init_encoder(args)
        if args.train_mode == "generative":
            self.generative_config = args.dataset_config["MAE"]
            self.init_decoder(args)

    def init_encoder(self, args):

        # step 0: define fourier basis STFT conv 
        self.loc_mod_stft = nn.ModuleDict()
        for loc in self.locations:
            self.loc_mod_stft[loc] = nn.ModuleDict()
            for mod in self.modalities:
                if type(self.config["loc_mod_conv_lens"]) is dict:
                    """for acoustic processing in Parkland data"""
                    conv_lens = self.config["loc_mod_conv_lens"][mod]
                    in_stride = self.config["loc_mod_in_conv_stride"][mod]
                else:
                    conv_lens = self.config["loc_mod_conv_lens"]
                    in_stride = 1
                self.loc_mod_stft[loc][mod] = FIC(self.args.dataset_config["loc_mod_spectrum_len"][loc][mod],stride=4) # TODO: check optimum stride?

        # Step 1: Single (loc, mod) feature
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
                    conv_lens=conv_lens,
                    dropout_ratio=self.config["dropout_ratio"],
                    num_inter_layers=self.config["loc_mod_conv_inter_layers"],
                    in_stride=in_stride,
                )

        # Step 3: Loc fusion
        self.loc_fusion_layers = nn.ModuleDict()
        self.mod_extractors = nn.ModuleDict()
        for mod in self.modalities:
            if self.self_attention:
                self.loc_fusion_layers[mod] = SelfAttentionFusionBlock()
            else:
                self.loc_fusion_layers[mod] = MeanFusionBlock()

            self.mod_extractors[mod] = ConvBlock(
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_mod_out_channels"],
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )

        # Step 5: GRU
        self.recurrent_layers = nn.ModuleDict()
        for mod in self.modalities:
            self.recurrent_layers[mod] = RecurrentBlock(
                in_channel=self.config["loc_out_channels"],
                out_channel=self.config["recurrent_dim"],
                num_layers=self.config["recurrent_layers"],
                dropout_ratio=self.config["dropout_ratio"],
            )

        # mod fusion layer, Cosmo --> attention fusion, MAE --> linear fusion
        if args.learn_framework == "Cosmo":
            "Attention fusion for Cosmo"
            self.cosmo_mod_fusion_layer = TransformerFusionBlock(
                self.config["recurrent_dim"] * 2,
                4,
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )
            self.sample_dim = self.config["recurrent_dim"] * 2
        elif args.learn_framework == "CMCV2":
            out_dim = self.args.dataset_config["CMCV2"]["emb_dim"]
            self.mod_projectors = nn.ModuleDict()
            for mod in self.modalities:
                self.mod_projectors[mod] = nn.Sequential(
                    nn.Linear(self.config["recurrent_dim"] * 2, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                )
            self.sample_dim = self.config["recurrent_dim"] * 2 * len(self.modalities)
        elif args.learn_framework == "MAE":
            """Linear fusion for MAE"""
            self.mae_mod_fusion_layer = nn.Sequential(
                nn.Linear(self.config["recurrent_dim"] * 2 * len(self.modalities), self.config["fc_dim"]),
                nn.GELU(),
                nn.Linear(self.config["fc_dim"], self.config["fc_dim"]),
            )
            self.sample_dim = self.config["fc_dim"]
        elif self.args.learn_framework == "GMC":
            """Projection head"""
            self.gmc_joint_projector = nn.Linear(
                self.config["recurrent_dim"] * 2 * len(self.modalities), self.config["recurrent_dim"] * 2
            )
            self.gmc_shared_projector = nn.Sequential(
                nn.Linear(self.config["recurrent_dim"] * 2, self.config["fc_dim"]),
                Swish(),
                nn.Linear(self.config["fc_dim"], self.config["fc_dim"]),
                Swish(),
                nn.Linear(self.config["fc_dim"], self.config["fc_dim"]),
            )
            self.sample_dim = self.config["recurrent_dim"] * 2 * len(self.modalities)
        else:
            self.sample_dim = self.config["recurrent_dim"] * 2 * len(self.modalities)

        # Classification layer
        if args.train_mode == "supervised" or self.config["pretrained_head"] == "linear":
            """Linear classification layers for supervised learning or finetuning."""
            self.class_layer = nn.Sequential(
                nn.Linear(self.sample_dim, args.dataset_config[args.task]["num_classes"]),
            )
        else:
            """Non-linear classification layers for self-supervised learning."""
            self.class_layer = nn.Sequential(
                nn.Linear(self.sample_dim, self.config["fc_dim"]),
                nn.GELU(),
                nn.Linear(self.config["fc_dim"], args.dataset_config[args.task]["num_classes"]),
            )

    def init_decoder(self, args):
        """
        Sample feature --> modality feature --> location feature --> input signal
        """
        # Masking token
        self.mask_token = nn.ModuleDict()
        for loc in self.locations:
            self.mask_token[loc] = nn.ParameterDict()
            for mod in self.modalities:
                self.mask_token[loc][mod] = nn.Parameter(
                    torch.zeros([1, 1, self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]])
                )

        # Step 0: Separate sample features into mod features
        self.mod_feature_sep_layer = nn.ModuleDict()
        for loc in self.locations:
            self.mod_feature_sep_layer[loc] = nn.ModuleDict()
            for mod in self.modalities:
                self.mod_feature_sep_layer[loc][mod] = nn.Sequential(
                    nn.Linear(self.config["fc_dim"], self.config["fc_dim"]),
                    nn.GELU(),  # remove last loss
                    nn.Linear(self.config["fc_dim"], self.config["recurrent_dim"] * 2),
                )

        # step 1: GRU decoder
        self.dec_recurrent_layers = nn.ModuleDict()
        for mod in self.modalities:
            self.dec_recurrent_layers[mod] = DecRecurrentBlock(
                mod_interval=self.args.dataset_config["num_segments"],
                in_channel=self.config["loc_out_channels"],
                out_channel=self.config["recurrent_dim"],
                num_layers=self.config["recurrent_layers"],
                dropout_ratio=self.config["dropout_ratio"],
            )
        # step 2: Loc fusion
        self.dec_loc_fusion_layers = nn.ModuleDict()
        self.dec_mod_extractors = nn.ModuleDict()
        for mod in self.modalities:
            if self.self_attention:
                self.dec_loc_fusion_layers[mod] = SelfAttentionFusionBlock()
            else:
                self.dec_loc_fusion_layers[mod] = MeanFusionBlock()

            self.dec_mod_extractors[mod] = DeConvBlock(
                num_segments=self.args.dataset_config["num_segments"],
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_mod_out_channels"],
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )
        # step 3: Single (loc, mod) feature decoder - DeConv Blocks
        self.dec_loc_mod_extractors = nn.ModuleDict()
        self.decoder_pred = nn.ModuleDict()
        for loc in self.locations:
            self.dec_loc_mod_extractors[loc] = nn.ModuleDict()
            self.decoder_pred[loc] = nn.ModuleDict()
            for mod in self.modalities:
                if type(self.config["loc_mod_conv_lens"]) is dict:
                    """for acoustic processing in Parkland data"""
                    conv_lens = self.config["loc_mod_conv_lens"][mod]
                    in_stride = self.config["loc_mod_in_conv_stride"][mod]
                else:
                    conv_lens = self.config["loc_mod_conv_lens"]
                    in_stride = 1

                # define the extractor
                self.dec_loc_mod_extractors[loc][mod] = DeConvBlock(
                    num_segments=self.args.dataset_config["num_segments"],
                    in_channels=args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    out_channels=self.config["loc_mod_out_channels"],
                    in_spectrum_len=args.dataset_config["loc_mod_spectrum_len"][loc][mod],
                    conv_lens=conv_lens,
                    dropout_ratio=self.config["dropout_ratio"],
                    num_inter_layers=self.config["loc_mod_conv_inter_layers"],
                    in_stride=in_stride,
                )

                input_dim = (
                    self.args.dataset_config["num_segments"]
                    * self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                    * self.args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                )
                input_dim = self.args.dataset_config["loc_mod_spectrum_len"][loc][mod]

                # Add another linear layer
                self.decoder_pred[loc][mod] = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.GELU(),
                    nn.Linear(input_dim, input_dim),
                )

    def process_input(self, freq_x, class_head):
        """
        Masking input for MAE
        Only mask for MAE pretraining
        """
        if self.args.train_mode != "generative" or class_head:
            return freq_x, None
        else:
            masked_x = {}
            masks = {}
            for loc in self.locations:
                masked_x[loc] = {}
                masks[loc] = {}
                for mod in self.modalities:
                    # mask ratio for each modality
                    mask_ratio = self.generative_config["masked_ratio"][mod]
                    b, c, i, s = freq_x[loc][mod].shape

                    patch_h, patch_w = (
                        self.generative_config["patch_size"][mod][0],
                        self.generative_config["patch_size"][mod][1],
                    )
                    patch_resolution_h, patch_resolution_w = i // patch_h, s // patch_w

                    masked_input, patch_mask, bit_mask = mask_input(
                        freq_x=freq_x[loc][mod],
                        input_resolution=(i, s),
                        patch_resolution=(i, s),
                        channel_dimension=-1,
                        window_size=(patch_h, patch_w),
                        mask_token=self.mask_token[loc][mod],
                        mask_ratio=mask_ratio,
                    )

                    masked_input = masked_input.reshape(b, c, i, s)
                    bit_mask = bit_mask.reshape(b, -1)

                    masked_x[loc][mod] = masked_input
                    masks[loc][mod] = bit_mask

            return (masked_x, masks)

    def forward_encoder(self, freq_x, class_head=True, proj_head=False):
        """The encoder function of DeepSense.
        Args:
            time_x (_type_): time_x is a dictionary consisting of the Tensor input of each input modality.
                        For each modality, the data is in (b, c (2 * 3 or 1), i (intervals), s (spectrum)) format.
        """

        # Step 1: Single (loc, mod) feature extraction, (b, c, i)
        loc_mod_features = {mod: [] for mod in self.modalities}
        for loc in self.locations:
            for mod in self.modalities:
                loc_mod_feature = self.loc_mod_extractors[loc][mod](freq_x[loc][mod])
                loc_mod_features[mod].append(loc_mod_feature)

        for mod in loc_mod_features:
            loc_mod_features[mod] = torch.stack(loc_mod_features[mod], dim=3)

        # Step 2: Location fusion, (b, c, i)
        mod_interval_features = {mod: [] for mod in self.modalities}
        for mod in self.modalities:
            if not self.multi_location_flag:
                mod_interval_features[mod] = loc_mod_features[mod].squeeze(3)
            else:
                fused_mod_feature = self.loc_fusion_layers[mod](loc_mod_features[mod])
                extracted_mod_features = self.mod_extractors[mod](fused_mod_feature)
                mod_interval_features[mod] = extracted_mod_features

        # Step 3: Interval Fusion for each modality, [b, c, i] --> [b, c]
        mod_features = []
        hidden_features = []
        for mod in self.modalities:
            mod_feature, hidden_feature = self.recurrent_layers[mod](mod_interval_features[mod])
            mod_features.append(mod_feature.flatten(start_dim=1))
            hidden_features.append(hidden_feature)

        # Step 4: Mod concatenation, [b, 1, mod, c]
        if not class_head:
            """Used in pretraining the framework"""
            if self.args.learn_framework == "MAE":
                """Return the merged sample features"""
                sample_features = self.forward_mae_fusion(mod_features)
                return sample_features, hidden_features
            elif self.args.learn_framework == "CMCV2":
                """Perform mod feature projection here."""
                if proj_head:
                    sample_features = {}
                    for i, mod in enumerate(self.modalities):
                        sample_features[mod] = self.mod_projectors[mod](mod_features[i])
                    return sample_features
                else:
                    return dict(zip(self.modalities, mod_features))
            elif self.args.learn_framework == "GMC":
                """Append a joint features that is the concated of all the mod views"""
                latent_features = []
                # individual mod features
                for i in range(len(self.modalities)):
                    latent_features.append(nn.functional.normalize(self.gmc_shared_projector(mod_features[i]), dim=-1))

                # joint features
                concated_features = torch.cat(mod_features, dim=1)
                joint_features = self.gmc_joint_projector(concated_features)

                # join together
                sample_features = dict(zip(self.modalities, latent_features))
                sample_features["joint"] = nn.functional.normalize(self.gmc_shared_projector(joint_features), dim=-1)
                return sample_features
            else:
                """CMC, Cosmo, and Cocoa, return the dict of mod features"""
                return dict(zip(self.modalities, mod_features))
        else:
            """Used in finetuning the classifier, return the sample features"""
            if self.args.learn_framework == "Cosmo":
                """Attention-based Fusion for Cosmo"""
                mod_features = torch.stack(mod_features, dim=1)
                mod_features = mod_features.unsqueeze(dim=1)
                sample_features = self.cosmo_mod_fusion_layer(mod_features).flatten(start_dim=1)
            elif self.args.learn_framework == "MAE":
                """FC fusion for MAE"""
                sample_features = self.forward_mae_fusion(mod_features)
            else:
                """Concatenation-based Fusion for CMC, CMCV2, Cocoa frameworks"""
                sample_features = torch.cat(mod_features, dim=1)

            logits = self.class_layer(sample_features)
            return logits

    def forward_mae_fusion(self, mod_features):
        """MAE fusion layer"""
        concat_mod_features = torch.cat(mod_features, dim=1)
        sample_features = self.mae_mod_fusion_layer(concat_mod_features)
        return sample_features

    def forward_decoder(self, sample_features, hidden_features):
        # Setep 0: Mod feature fusion (concat + fc --> [b, c])--> Mod feature separation
        sep_mod_features = []
        for loc in self.locations:
            for mod in self.modalities:
                sep_mod_feature = self.mod_feature_sep_layer[loc][mod](sample_features)
                sep_mod_features.append(sep_mod_feature)

        # Step 1: Interval Fusion Decoder from each modality, [b, c, i]
        dec_mod_features = {}
        for i, mod in enumerate(self.modalities):
            dec_mod_feature = self.dec_recurrent_layers[mod](sep_mod_features[i], hidden_features[i])
            dec_mod_features[mod] = dec_mod_feature

        # Step 2: Location fusion decoder, (b, c, i)
        dec_mod_interval_features = {}
        for mod in self.modalities:
            if not self.multi_location_flag:
                dec_mod_interval_features[mod] = dec_mod_features[mod].unsqueeze(3)
            else:
                # TODO:
                fused_mod_feature = self.dec_loc_fusion_layers[mod](dec_mod_features[mod])
                extracted_mod_features = self.dec_mod_extractors[mod](fused_mod_feature)
                dec_mod_interval_features[mod] = extracted_mod_features

        # for mod in self.modalities:
        # TODO: Stack -> UnStack
        # dec_mod_interval_features[mod] = dec_mod_interval_features[mod].squeeze(3)

        # Step 3: Single (loc, mod) feature extraction, (b, c, i)
        dec_loc_mod_input = {}
        for i, loc in enumerate(self.locations):
            dec_loc_mod_input[loc] = {}
            for mod in self.modalities:
                decoded_input = self.dec_loc_mod_extractors[loc][mod](dec_mod_interval_features[mod][:, :, :, i])
                decoded_input = self.decoder_pred[loc][mod](decoded_input)
                dec_loc_mod_input[loc][mod] = decoded_input

        return dec_loc_mod_input


    def forward(self, freq_x, class_head=True, decoding=True):
        
        if self.args.fic:
            # step 0 - process input with custom stft
            for loc in self.locations:
                for mod in self.modalities:
                    freq_x[loc][mod] = self.loc_mod_stft[loc][mod](freq_x[loc][mod])

        processed_freq_x, masks = self.process_input(freq_x, class_head)

        if class_head:
            """Finetuning the classifier"""
            logits = self.forward_encoder(processed_freq_x, class_head)
            return logits
        else:
            """Pretraining the framework"""
            if self.args.train_mode != "generative":
                """CMC, Cosmo, Cocoa"""
                enc_mod_features = self.forward_encoder(processed_freq_x, class_head, proj_head)
                return enc_mod_features
            else:
                """MAE"""
                # Encoding
                enc_sample_features, hidden_features = self.forward_encoder(processed_freq_x, class_head)

                if not decoding:
                    """Used in KNN eval"""
                    return enc_sample_features
                else:
                    # Decoding
                    dec_output = self.forward_decoder(enc_sample_features, hidden_features)

                    return dec_output, freq_x, masks, enc_sample_features

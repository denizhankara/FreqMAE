import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn import TransformerEncoderLayer
from input_utils.padding_utils import get_padded_size
from input_utils.mask_utils import mask_input, cross_coherence
from models.FusionModules import TransformerFusionBlock
from models.ConvModules import FIC
from timm.models.layers import trunc_normal_
from models.GMCModules import Swish

from models.SwinModules import (
    BasicLayer,
    PatchEmbed,
    PatchExpanding,
    PatchMerging,
)


class TransformerV4_CMC(nn.Module):
    """
    SWIN Transformer model

    Parameters
    ----------

    Returns
    -------
    int or float
        The square of `x`
    """

    def __init__(self, args) -> None:
        """
        SWIN Transformer model constructor

        Parameters
        ----------
        args:
            list of configuration
        """

        super().__init__()
        self.args = args
        self.config = args.dataset_config["TransformerV4"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_segments = args.dataset_config["num_segments"]

        # Transformer Variables
        self.drop_rate = self.config["dropout_ratio"]
        self.norm_layer = nn.LayerNorm
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.init_encoder()
        if self.generative_config["mask_scheme_finetune"] == "attention":
            self.initialize_attentions() # new, for randomizing attentions

        if args.train_mode == "generative":
            self.generative_config = args.dataset_config[args.learn_framework]
            self.init_decoder()

    def init_encoder(self) -> None:
        """
        Single (mod, loc) feature extraction --> loc fusion --> mod fusion
        """
        # step 0: define fourier basis STFT conv 
        self.loc_mod_stft = nn.ModuleDict()
        for loc in self.locations:
            self.loc_mod_stft[loc] = nn.ModuleDict()
            for mod in self.modalities:
                self.loc_mod_stft[loc][mod] = FIC(self.args.dataset_config["loc_mod_spectrum_len"][loc][mod],stride=4) # TODO: check optimum stride?
        self.generative_config = self.args.dataset_config[self.args.learn_framework]
        self.masked_ratio_finetune = self.generative_config["masked_ratio_finetune"]
        # Single sensor
        self.freq_interval_layers = nn.ModuleDict()  # Frequency & Interval layers
        self.patch_embed = nn.ModuleDict()
        self.absolute_pos_embed = nn.ModuleDict()
        self.mod_patch_embed = nn.ModuleDict()
        self.mod_in_layers = nn.ModuleDict()

        self.layer_dims = {}
        self.img_sizes = {}
        
        self.cross_fusing_attention_v2 = CrossAttention_v2(nhead=self.config["time_freq_head_num"], d_model = self.config["mod_out_channels"])
        self.cross_fusing_attention_v3 = CrossAttention_v3(nhead=self.config["time_freq_head_num"], d_model = self.config["mod_out_channels"])

        for loc in self.locations:
            self.freq_interval_layers[loc] = nn.ModuleDict()
            self.patch_embed[loc] = nn.ModuleDict()
            self.absolute_pos_embed[loc] = nn.ParameterDict()
            self.mod_in_layers[loc] = nn.ModuleDict()

            self.layer_dims[loc] = {}
            self.img_sizes[loc] = {}
            for mod in self.modalities:
                # Decide the spatial size for "image"
                stride = self.config["in_stride"][mod]
                spectrum_len = self.args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                img_size = (self.num_segments, spectrum_len // stride)

                # get the padded image size
                padded_img_size = get_padded_size(
                    img_size,
                    self.config["window_size"][mod],
                    self.config["patch_size"]["freq"][mod],
                    len(self.config["time_freq_block_num"][mod]),
                )
                self.img_sizes[loc][mod] = padded_img_size

                # Patch embedding and Linear embedding (H, W, in_channel) -> (H / p_size, W / p_size, C)
                self.patch_embed[loc][mod] = PatchEmbed(
                    img_size=padded_img_size,
                    patch_size=self.config["patch_size"]["freq"][mod],
                    in_chans=self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod] * stride,
                    embed_dim=self.config["time_freq_out_channels"],
                    norm_layer=self.norm_layer,
                )
                patches_resolution = self.patch_embed[loc][mod].patches_resolution

                # Absolute positional embedding (optional)
                self.absolute_pos_embed[loc][mod] = nn.Parameter(
                    torch.zeros(1, self.patch_embed[loc][mod].num_patches, self.config["time_freq_out_channels"])
                )
                trunc_normal_(self.absolute_pos_embed[loc][mod], std=0.02)

                # Swin Transformer Block
                self.freq_interval_layers[loc][mod] = nn.ModuleList()

                # Drop path rate
                dpr = [
                    x.item()
                    for x in torch.linspace(
                        0, self.config["drop_path_rate"], sum(self.config["time_freq_block_num"][mod])
                    )
                ]  # stochastic depth decay rule

                for i_layer, block_num in enumerate(
                    self.config["time_freq_block_num"][mod]
                ):  # different downsample ratios
                    down_ratio = 2**i_layer
                    layer_dim = int(self.config["time_freq_out_channels"] * down_ratio)
                    layer = BasicLayer(
                        dim=layer_dim,  # C in SWIN
                        input_resolution=(
                            patches_resolution[0] // down_ratio,  # Patch resolution = (H/4, W/4)
                            patches_resolution[1] // down_ratio,
                        ),
                        num_heads=self.config["time_freq_head_num"],
                        window_size=self.config["window_size"][mod].copy(),
                        depth=block_num,
                        drop=self.drop_rate,
                        attn_drop=self.config["attn_drop_rate"],
                        drop_path=dpr[
                            sum(self.config["time_freq_block_num"][mod][:i_layer]) : sum(
                                self.config["time_freq_block_num"][mod][: i_layer + 1]
                            )
                        ],
                        norm_layer=self.norm_layer,
                        downsample=PatchMerging
                        if (i_layer < len(self.config["time_freq_block_num"][mod]) - 1)
                        else None,
                    )
                    self.freq_interval_layers[loc][mod].append(layer)

                # Unify the input channels for each modality
                self.mod_in_layers[loc][mod] = nn.Linear(
                    (patches_resolution[0] // down_ratio) * (patches_resolution[1] // down_ratio) * layer_dim,
                    self.config["loc_out_channels"],
                )

        # Loc fusion, [b, i, c], loc contextual feature extraction + loc fusion
        if len(self.locations) > 1:
            self.loc_context_layers = nn.ModuleDict()
            self.loc_fusion_layer = nn.ModuleDict()
            for mod in self.modalities:
                """Single mod contextual feature extraction"""
                module_list = [
                    TransformerEncoderLayer(
                        d_model=self.config["loc_out_channels"],
                        nhead=self.config["loc_head_num"],
                        dim_feedforward=self.config["loc_out_channels"],
                        dropout=self.config["dropout_ratio"],
                        batch_first=True,
                    )
                    for _ in range(self.config["loc_block_num"])
                ]
                self.loc_context_layers[mod] = nn.Sequential(*module_list)

                """Loc fusion layer for each mod"""
                self.loc_fusion_layer[mod] = TransformerFusionBlock(
                    self.config["loc_out_channels"],
                    self.config["loc_head_num"],
                    self.config["dropout_ratio"],
                    self.config["dropout_ratio"],
                )

        # mod fusion layer
        if self.args.learn_framework == "Cosmo":
            "Attention fusion for Cosmo"
            self.cosmo_mod_fusion_layer = TransformerFusionBlock(
                self.config["loc_out_channels"],
                self.config["loc_head_num"],
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )
            self.sample_dim = self.config["loc_out_channels"]
        elif self.args.learn_framework == "CMCV2":
            """Mod feature projection, and attention fusion."""
            out_dim = self.args.dataset_config["CMCV2"]["emb_dim"]
            self.mod_projectors = nn.ModuleDict()
            for mod in self.modalities:
                self.mod_projectors[mod] = nn.Sequential(
                    nn.Linear(self.config["loc_out_channels"], out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                )
            self.cosmo_mod_fusion_layer = TransformerFusionBlock(
                self.config["loc_out_channels"],
                self.config["loc_head_num"],
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )
            self.sample_dim = self.config["loc_out_channels"]
        elif self.args.learn_framework == "MAE":
            """Linear fusion for MAE"""
            self.mae_mod_fusion_layers = nn.ModuleDict()
            for mod in self.modalities:
                self.mae_mod_fusion_layers[mod] = nn.Sequential(
                    nn.Linear(self.config["loc_out_channels"] * (len(self.modalities)-1) , self.config["fc_dim"]),
                    nn.GELU(),
                    nn.Linear(self.config["fc_dim"], self.config["loc_out_channels"]),
                )
            # self.mae_mod_fusion_layer = nn.Sequential(
            #     nn.Linear(self.config["loc_out_channels"] * (len(self.modalities)-1), self.config["fc_dim"]),
            #     nn.GELU(),
            #     nn.Linear(self.config["fc_dim"], self.config["loc_out_channels"]),
            # )
            self.sample_dim = self.config["fc_dim"]
        elif self.args.learn_framework == "GMC":
            """Projection head"""
            self.gmc_joint_projector = nn.Linear(
                self.config["loc_out_channels"] * len(self.modalities), self.config["loc_out_channels"]
            )
            self.gmc_shared_projector = nn.Sequential(
                nn.Linear(self.config["loc_out_channels"], self.config["fc_dim"]),
                Swish(),
                nn.Linear(self.config["fc_dim"], self.config["fc_dim"]),
                Swish(),
                nn.Linear(self.config["fc_dim"], self.config["fc_dim"]),
            )
            self.sample_dim = self.config["loc_out_channels"] * len(self.modalities)
        else:
            self.sample_dim = self.config["loc_out_channels"] * len(self.modalities)

        # Classification layer
        if self.args.train_mode == "supervised" or self.config["pretrained_head"] == "linear":
            """Linear classification layers for supervised learning or finetuning."""
            self.class_layer = nn.Sequential(
                nn.Linear(len(self.modalities) * self.config["loc_out_channels"] + len(self.modalities) * self.config["loc_out_channels"], self.args.dataset_config[self.args.task]["num_classes"]),
            )
            
        else:
            """Non-linear classification layers for self-supervised learning."""
            self.class_layer = nn.Sequential(
                nn.Linear(self.sample_dim, self.config["fc_dim"]),
                nn.GELU(),
                nn.Linear(self.config["fc_dim"], self.args.dataset_config[self.args.task]["num_classes"]),
            )

    def init_decoder(self) -> None:
        """
        Sample feature --> modality feature --> location feature --> input signal
        """
        # Step 0: Separate sample features into mod features
        self.mod_feature_sep_layer = nn.ModuleDict()
        for loc in self.locations:
            self.mod_feature_sep_layer[loc] = nn.ModuleDict()
            for mod in self.modalities:
                self.mod_feature_sep_layer[loc][mod] = nn.Sequential(
                    nn.Linear(self.config["fc_dim"], self.config["fc_dim"]),
                    nn.GELU(),
                    nn.Linear(self.config["fc_dim"], self.config["loc_out_channels"]),
                )

        self.mask_token = nn.ModuleDict()
        self.masked_ratio = self.generative_config["masked_ratio"]

        self.patch_expand = nn.ModuleDict()
        self.decoder_blocks = nn.ModuleDict()
        self.decoder_pred = nn.ModuleDict()
        self.mod_out_layers = nn.ModuleDict()


        for loc in self.locations:
            self.patch_expand[loc] = nn.ModuleDict()
            self.mask_token[loc] = nn.ParameterDict()
            self.decoder_blocks[loc] = nn.ModuleDict()
            self.decoder_pred[loc] = nn.ModuleDict()
            self.mod_out_layers[loc] = nn.ModuleDict()

            for mod in self.modalities:
                self.mask_token[loc][mod] = nn.Parameter(torch.zeros([1, 1, self.config["time_freq_out_channels"]]))
                self.patch_expand[loc][mod] = PatchExpanding(
                    self.img_sizes[loc][mod],
                    embed_dim=self.config["time_freq_out_channels"]
                    * 2 ** (len(self.config["time_freq_block_num"][mod]) - 1),
                    norm_layer=self.norm_layer,
                )
                self.decoder_blocks[loc][mod] = nn.ModuleList()
                patches_resolution = self.patch_embed[loc][mod].patches_resolution

                # Drop path rate
                dpr = [
                    x.item()
                    for x in torch.linspace(
                        0, self.config["drop_path_rate"], sum(self.config["time_freq_block_num"][mod])
                    )
                ]  # stochastic depth decay rule

                for i_layer, block_num in enumerate(self.config["time_freq_block_num"][mod][:-1]):
                    inverse_i_layer = len(self.config["time_freq_block_num"][mod]) - i_layer - 2
                    down_ratio = 2**inverse_i_layer
                    layer_dim = int(self.config["time_freq_out_channels"] * down_ratio)
                    layer = BasicLayer(
                        dim=layer_dim,  # C in SWIN
                        input_resolution=(
                            patches_resolution[0] // down_ratio,  # Patch resolution = (H/4, W/4)
                            patches_resolution[1] // down_ratio,
                        ),
                        num_heads=self.config["time_freq_head_num"],
                        window_size=self.config["window_size"][mod].copy(),
                        depth=block_num,
                        drop=self.drop_rate,
                        attn_drop=self.config["attn_drop_rate"],
                        drop_path=dpr[
                            sum(self.config["time_freq_block_num"][mod][:inverse_i_layer]) : sum(
                                self.config["time_freq_block_num"][mod][: inverse_i_layer + 1]
                            )
                        ],
                        norm_layer=self.norm_layer,
                        patch_expanding=PatchExpanding
                        if (i_layer < len(self.config["time_freq_block_num"][mod]) - 2)
                        else None,
                    )
                    self.decoder_blocks[loc][mod].append(layer)

                down_ratio = 2 ** (len(self.config["time_freq_block_num"][mod]) - 1)
                layer_dim = int(self.config["time_freq_out_channels"] * down_ratio)
                self.layer_dims[loc][mod] = layer_dim

                self.mod_out_layers[loc][mod] = nn.Linear(
                    self.config["loc_out_channels"],
                    (patches_resolution[0] // down_ratio) * (patches_resolution[1] // down_ratio) * layer_dim,
                )

                patch_area = self.config["patch_size"]["freq"][mod][0] * self.config["patch_size"]["freq"][mod][1]

                self.decoder_pred[loc][mod] = nn.Sequential(
                    nn.Linear(self.config["time_freq_out_channels"], self.config["time_freq_out_channels"]),
                    nn.GELU(),
                    nn.Linear(
                        self.config["time_freq_out_channels"],
                        patch_area * self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    ),
                )

        self.patch_expand_fused = nn.ModuleDict()
        self.decoder_blocks_fused = nn.ModuleDict()
        self.decoder_pred_fused = nn.ModuleDict()
        self.mod_out_layers_fused = nn.ModuleDict()


        for loc in self.locations:
            self.patch_expand_fused[loc] = nn.ModuleDict()
            self.decoder_blocks_fused[loc] = nn.ModuleDict()
            self.decoder_pred_fused[loc] = nn.ModuleDict()
            self.mod_out_layers_fused[loc] = nn.ModuleDict()

            for mod in self.modalities:
                self.mask_token[loc][mod] = nn.Parameter(torch.zeros([1, 1, self.config["time_freq_out_channels"]]))
                self.patch_expand_fused[loc][mod] = PatchExpanding(
                    self.img_sizes[loc][mod],
                    embed_dim=self.config["time_freq_out_channels"]
                    * 2 ** (len(self.config["time_freq_block_num"][mod]) - 1),
                    norm_layer=self.norm_layer,
                )
                self.decoder_blocks_fused[loc][mod] = nn.ModuleList()
                patches_resolution = self.patch_embed[loc][mod].patches_resolution

                # Drop path rate
                dpr = [
                    x.item()
                    for x in torch.linspace(
                        0, self.config["drop_path_rate"], sum(self.config["time_freq_block_num"][mod])
                    )
                ]  # stochastic depth decay rule

                for i_layer, block_num in enumerate(self.config["time_freq_block_num"][mod][:-1]):
                    inverse_i_layer = len(self.config["time_freq_block_num"][mod]) - i_layer - 2
                    down_ratio = 2**inverse_i_layer
                    layer_dim = int(self.config["time_freq_out_channels"] * down_ratio)
                    layer = BasicLayer(
                        dim=layer_dim,  # C in SWIN
                        input_resolution=(
                            patches_resolution[0] // down_ratio,  # Patch resolution = (H/4, W/4)
                            patches_resolution[1] // down_ratio,
                        ),
                        num_heads=self.config["time_freq_head_num"],
                        window_size=self.config["window_size"][mod].copy(),
                        depth=block_num,
                        drop=self.drop_rate,
                        attn_drop=self.config["attn_drop_rate"],
                        drop_path=dpr[
                            sum(self.config["time_freq_block_num"][mod][:inverse_i_layer]) : sum(
                                self.config["time_freq_block_num"][mod][: inverse_i_layer + 1]
                            )
                        ],
                        norm_layer=self.norm_layer,
                        patch_expanding=PatchExpanding
                        if (i_layer < len(self.config["time_freq_block_num"][mod]) - 2)
                        else None,
                    )
                    self.decoder_blocks_fused[loc][mod].append(layer)

                down_ratio = 2 ** (len(self.config["time_freq_block_num"][mod]) - 1)
                layer_dim = int(self.config["time_freq_out_channels"] * down_ratio)
                self.layer_dims[loc][mod] = layer_dim

                self.mod_out_layers_fused[loc][mod] = nn.Linear(
                    self.config["loc_out_channels"],
                    (patches_resolution[0] // down_ratio) * (patches_resolution[1] // down_ratio) * layer_dim,
                )

                patch_area = self.config["patch_size"]["freq"][mod][0] * self.config["patch_size"]["freq"][mod][1]

                self.decoder_pred_fused[loc][mod] = nn.Sequential(
                    nn.Linear(self.config["time_freq_out_channels"], self.config["time_freq_out_channels"]),
                    nn.GELU(),
                    nn.Linear(
                        self.config["time_freq_out_channels"],
                        patch_area * self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    ),
                )

    def pad_input(self, freq_x, loc, mod):
        stride = self.config["in_stride"][mod]
        spectrum_len = self.args.dataset_config["loc_mod_spectrum_len"][loc][mod]
        img_size = (self.num_segments, spectrum_len // stride)
        freq_input = freq_x[loc][mod]

        # [b, c, i, spectrum] -- > [b, i, spectrum, c]
        freq_input = torch.permute(freq_input, [0, 2, 3, 1])
        b, i, s, c = freq_input.shape

        # Forces both audio and seismic to have the same "img" size
        freq_input = torch.reshape(freq_input, (b, i, s // stride, c * stride))

        # Repermute back to [b, c, i, spectrum], (b, c, h, w) required in PatchEmbed
        freq_input = torch.permute(freq_input, [0, 3, 1, 2])

        # Pad [i, spectrum] to the required padding size
        padded_img_size = self.patch_embed[loc][mod].img_size
        padded_height = padded_img_size[0] - img_size[0]
        padded_width = padded_img_size[1] - img_size[1]

        # test different padding
        freq_input = F.pad(input=freq_input, pad=(0, padded_width, 0, padded_height), mode="constant", value=0)

        return freq_input, padded_img_size

    def forward_encoder(self, patched_input, class_head=True, proj_head=False):
        """
        If class_head is False, we return the modality features; otherwise, we return the classification results.
        time-freq feature extraction --> loc fusion --> mod concatenation --> class layer
        """
        # Step 1: Feature extractions on time interval (i) and spectrum (s) domains
        mod_loc_features = {mod: [] for mod in self.modalities}
        for loc in self.locations:
            for mod in self.modalities:
                embeded_input = patched_input[loc][mod]
                b = embeded_input.shape[0]

                # Absolute positional embedding
                if self.config["APE"]:
                    embeded_input = embeded_input + self.absolute_pos_embed[loc][mod]

                # SwinTransformer Layer block
                for layer in self.freq_interval_layers[loc][mod]:
                    freq_interval_output = layer(embeded_input)
                    embeded_input = freq_interval_output

                # Unify the input channels for each modality
                freq_interval_output = self.mod_in_layers[loc][mod](freq_interval_output.reshape([b, -1]))
                freq_interval_output = freq_interval_output.reshape(b, 1, -1)

                # Append the modality feature to the list
                mod_loc_features[mod].append(freq_interval_output)

        # Concatenate the location features, [b, i, location, c]
        for mod in self.modalities:
            mod_loc_features[mod] = torch.stack(mod_loc_features[mod], dim=2)

        # Step 2: Loc feature fusion and extraction for each mod, [b, i, location, c]
        mod_features = []
        for mod in mod_loc_features:
            if len(self.locations) > 1:
                """Extract mod feature with peer-feature context"""
                b, i, locs, c = mod_loc_features[mod].shape
                mod_loc_input = mod_loc_features[mod].reshape([b * i, locs, c])
                mod_loc_context_feature = self.loc_context_layers[mod](mod_loc_input)
                mod_loc_context_feature = mod_loc_context_feature.reshape([b, i, locs, c])

                """Mod feature fusion, [b, 1, 1, c] -- > [b, c]"""
                mod_feature = self.loc_fusion_layer[mod](mod_loc_context_feature)
                mod_feature = mod_feature.flatten(start_dim=1)
                mod_features.append(mod_feature)
            else:
                mod_features.append(mod_loc_features[mod].flatten(start_dim=1))

        # Step 3: Mod concatenation, [b, 1, mod, c]
        if not class_head:
            if self.args.learn_framework == "MAE":
                """Return the merged sample features"""
                sample_features, fused_features = self.forward_mae_fusion(mod_features)
                return sample_features, fused_features
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
                    latent_features.append(F.normalize(self.gmc_shared_projector(mod_features[i]), dim=-1))

                # joint features
                concated_features = torch.cat(mod_features, dim=1)
                joint_features = self.gmc_joint_projector(concated_features)

                # join together
                sample_features = dict(zip(self.modalities, latent_features))
                sample_features["joint"] = F.normalize(self.gmc_shared_projector(joint_features), dim=-1)
                return sample_features
            else:
                """CMC, Cosmo, and Cocoa, return the dict of mod features"""
                return dict(zip(self.modalities, mod_features))
        else:
            if self.args.learn_framework == "Cosmo":
                """Attention-based fusion."""
                mod_features = torch.stack(mod_features, dim=1)
                mod_features = mod_features.unsqueeze(dim=1)
                sample_features = self.cosmo_mod_fusion_layer(mod_features).flatten(start_dim=1)
            elif self.args.learn_framework == "CMCV2":
                """Mod feature projection and attention-based fusion"""
                mod_features = torch.stack(mod_features, dim=1)
                mod_features = mod_features.unsqueeze(dim=1)
                sample_features = self.cosmo_mod_fusion_layer(mod_features).flatten(start_dim=1)
            elif self.args.learn_framework == "MAE":
                """FC fusion for MAE"""
                sample_features, fused_features = self.forward_mae_fusion(mod_features)
                # concatenate the fused features and sample features
                sample_features = torch.cat([sample_features, fused_features], dim=1)
            else:
                """Concatenation-based fusion."""
                sample_features = torch.cat(mod_features, dim=1)

            logits = self.class_layer(sample_features)
            return logits

    def forward_mae_fusion(self, mod_features):
        """MAE linear fusion layer"""
        # concat_mod_features = torch.cat(mod_features, dim=1)
        # sample_features = self.mae_mod_fusion_layer(concat_mod_features)
        # return sample_features

        
        """MAE embedding masking based fusion"""
        # embeddings masking
        fused_features = []
        
        for mod_index, mod_feature in enumerate(mod_features):
            masked_mod = self.modalities[mod_index]
            # concatenate all the other modalities\
            other_mod_features = torch.cat([mod_features[i] for i in range(len(mod_features)) if i != mod_index], dim=1)
            # fuse the other modalities
            other_mod_fusion = self.mae_mod_fusion_layers[masked_mod](other_mod_features)
            # add to fused features
            fused_features.append(other_mod_fusion)
            
        # concatenate the fused features
        fused_features = torch.cat(fused_features, dim=1)
        
        # concatenate the mod features
        mod_features = torch.cat(mod_features, dim=1)
        return mod_features, fused_features 
            
        
        
        """MAE attention based fusion scheme"""
        # attention-based fusion
        # sample_features = self.forward_mae_fusion_cross_attention(mod_features)
        # sample_features = self.mae_mod_fusion_layer(sample_features)
        # return sample_features
    
        # attention-fusion v2
        # sample_features = self.cross_fusing_attention_v2(mod_features)
        # sample_features = self.mae_mod_fusion_layer(sample_features)
        # return sample_features
        
        
        # attention-fusion v3 - Works better than v2, close to original
        # sample_features = self.cross_fusing_attention_v3(mod_features)
        # sample_features = self.mae_mod_fusion_layer(sample_features)
        # return sample_features
        

    
    def forward_mae_fusion_cross_attention(self, encoded_mod_features):
        
        B, C = encoded_mod_features[0].shape
        N = len(encoded_mod_features)
        tensors = encoded_mod_features
        # Step 1: Calculate attention weights
        attention_weights = torch.zeros(B, B).to(encoded_mod_features[0].device)
        for i in range(N):
            for j in range(N):
                attention_weights += torch.matmul(tensors[i], tensors[j].t())

        # Step 2: Normalize attention weights
        attention_weights = F.softmax(attention_weights, dim=1)

        # Step 3: Apply attention weights to input tensors
        attended_tensors = []
        for i in range(N):
            attended_tensor = torch.matmul(attention_weights, tensors[i])
            attended_tensors.append(attended_tensor)

        # Step 4: Concatenate attended tensors
        fused_embedding = torch.cat(attended_tensors, dim=1)

        return fused_embedding

    def forward_decoder(self, sample_features,fused_features):
        """
        Decode the latent features
        """
        # Setep 0: Mod feature separation
        # sep_mod_features = []
        # for loc in self.locations:
        #     for mod in self.modalities:
        #         sep_mod_feature = self.mod_feature_sep_layer[loc][mod](sample_features)
        #         sep_mod_features.append(sep_mod_feature)
        
        # mod feature separation
        sep_mod_features = []
        sep_fused_features = []
        i = 0
        for loc in self.locations:
            for mod in self.modalities:
                # direct separation
                current_index = i + self.config["loc_out_channels"]
                sep_mod_feature = sample_features[:, i:current_index]
                sep_fused_feature = fused_features[:, i:current_index]
                i = current_index
                sep_mod_features.append(sep_mod_feature)
                sep_fused_features.append(sep_fused_feature)
                
        
        # for sep mod features
        decoded_out = {}
        for loc in self.locations:
            decoded_out[loc] = {}
            for i, mod in enumerate(self.modalities):
                # Expand channels --> Rebuild spatial dimensions
                decoded_out[loc][mod] = []
                decoded_mod_feature = self.mod_out_layers[loc][mod](sep_mod_features[i])
                decoded_mod_feature = decoded_mod_feature.reshape(
                    decoded_mod_feature.shape[0], -1, self.layer_dims[loc][mod]
                )
                decoded_mod_feature = self.patch_expand[loc][mod](decoded_mod_feature)

                # Decoder blocks
                for layer in self.decoder_blocks[loc][mod]:
                    decoded_tokens = layer(decoded_mod_feature)
                    decoded_mod_feature = decoded_tokens

                # Prediction layer with FCs
                decoded_tokens = self.decoder_pred[loc][mod](decoded_tokens)
                decoded_out[loc][mod] = decoded_tokens
        
        
        # for sep fused features
        decoded_out_fused = {}
        for loc in self.locations:
            decoded_out_fused[loc] = {}
            for i, mod in enumerate(self.modalities):
                # Expand channels --> Rebuild spatial dimensions
                decoded_out_fused[loc][mod] = []
                decoded_mod_feature = self.mod_out_layers_fused[loc][mod](sep_fused_features[i])
                decoded_mod_feature = decoded_mod_feature.reshape(
                    decoded_mod_feature.shape[0], -1, self.layer_dims[loc][mod]
                )
                decoded_mod_feature = self.patch_expand_fused[loc][mod](decoded_mod_feature)

                # Decoder blocks
                for layer in self.decoder_blocks_fused[loc][mod]:
                    decoded_tokens = layer(decoded_mod_feature)
                    decoded_mod_feature = decoded_tokens

                # Prediction layer with FCs
                decoded_tokens = self.decoder_pred_fused[loc][mod](decoded_tokens)
                decoded_out_fused[loc][mod] = decoded_tokens

        return decoded_out, decoded_out_fused

    def patch_forward(self, freq_x, class_head=True):
        """Patch the input and mask for pretrianing

        Args:
            freq_x (_type_): [loc][mod] data
            class_head (bool, optional): whether to include classification layer. Defaults to True.

        Returns:
            [loc][mod]: patched input
        """
        embeded_inputs = {}
        padded_inputs = {}
        mod_loc_masks = {}
        for loc in self.locations:
            embeded_inputs[loc] = {}
            padded_inputs[loc] = {}
            mod_loc_masks[loc] = {}
            for mod in self.modalities:
                # Pad the input and store the padded input
                freq_input, padded_img_size = self.pad_input(freq_x, loc, mod)
                padded_inputs[loc][mod] = freq_input

                if (self.args.stage=="pretrain" and self.generative_config["mask_scheme"] == "coherence") or (self.args.stage == "finetune" and self.generative_config["mask_scheme_finetune"] == "coherence"):
                    # spectral coherence based mask selection
                    spectral_coherence = cross_coherence(freq_input)
                    spectral_coherence = torch.nan_to_num(spectral_coherence, nan=1)
                else:
                    spectral_coherence = None

                # Patch Partition and Linear Embedding
                embeded_input = self.patch_embed[loc][mod](freq_input)

                # we only mask images for pretraining MAE
                if self.args.train_mode == "generative" and class_head == False:
                    embeded_input, mod_loc_mask, mod_loc_bit_mask = mask_input(
                        freq_x=embeded_input,
                        input_resolution=padded_img_size,
                        patch_resolution=self.patch_embed[loc][mod].patches_resolution,
                        channel_dimension=-1,
                        window_size=self.config["window_size"][mod],
                        mask_token=self.mask_token[loc][mod],
                        mask_ratio=self.masked_ratio[mod],
                        mask_scheme = self.generative_config["mask_scheme"],
                        spectral_coherence=spectral_coherence,
                    )
                    # embeded_input, mod_loc_mask = window_masking(
                    #     embeded_input,
                    #     input_resolution=padded_img_size,
                    #     patch_resolution=self.patch_embed[loc][mod].patches_resolution,
                    #     window_size=self.config["window_size"][mod],
                    #     mask_token=self.mask_token[loc][mod],
                    #     mask_ratio=self.masked_ratio[mod],
                    # )
                    mod_loc_masks[loc][mod] = mod_loc_mask

                # light-weight mask with task-specific structure for finetuning
                elif self.args.train_mode == "generative" and self.args.stage=="finetune" and self.generative_config["mask_scheme_finetune"] != "none":
                    embeded_input, mod_loc_mask, mod_loc_bit_mask = mask_input(
                        freq_x=embeded_input,
                        input_resolution=padded_img_size,
                        patch_resolution=self.patch_embed[loc][mod].patches_resolution,
                        channel_dimension=-1,
                        window_size=self.config["window_size"][mod],
                        mask_token=self.mask_token[loc][mod],
                        mask_ratio=self.masked_ratio_finetune[mod],
                        mask_scheme = self.generative_config["mask_scheme_finetune"],
                        spectral_coherence=spectral_coherence,
                    )
                        
                embeded_inputs[loc][mod] = embeded_input
                        
        return embeded_inputs, padded_inputs, mod_loc_masks


    def forward(self, freq_x, class_head=True, decoding=True):
        if self.args.fic:
            # step 0 - process input with custom stft
            for loc in self.locations:
                for mod in self.modalities:
                    freq_x[loc][mod] = self.loc_mod_stft[loc][mod](freq_x[loc][mod])

        # PatchEmbed the input, window mask if MAE
        patched_inputs, padded_inputs, masks = self.patch_forward(freq_x, class_head)

        if class_head:
            """Finetuning the classifier"""
            logits = self.forward_encoder(patched_inputs, class_head)
            if self.generative_config["mask_scheme_finetune"] == "attention":
                self.collect_attentions() # new
            return logits
        else:
            """Pretraining the framework"""
            if self.args.train_mode != "generative":
                """CMC, Cosmo, Cocoa"""
                enc_mod_features = self.forward_encoder(patched_inputs, class_head, proj_head)
                return enc_mod_features
            else:
                enc_sample_features, enc_fused_features = self.forward_encoder(patched_inputs, class_head)
                if not decoding:
                    """Used in KNN eval"""
                    enc_sample_features = torch.cat([enc_sample_features, enc_fused_features], dim=1)
                    return enc_sample_features
                else:
                    # Decoding
                    dec_output_sample, dec_output_fused = self.forward_decoder(enc_sample_features,enc_fused_features)

                return dec_output_sample, dec_output_fused, padded_inputs, masks, enc_sample_features

    def collect_attentions(self):
        """Collect attention maps from the model"""
        all_attentions = {}
        for loc in self.locations:
            all_attentions[loc] = {}
            for mod in self.modalities:
                all_attentions[loc][mod] = []
                layers = self.freq_interval_layers[loc][mod]
                for basiclayer in layers: # 3 basic layers
                    for block in basiclayer.blocks: # 2 blocks
                        block_attentions = block.attentions
                        all_attentions[loc][mod].append(block_attentions) # block_attentions[0].shape = (16384, 4, 4, 4)
        
        # set new attentions
        self.all_attentions = all_attentions
    
    def initialize_attentions(self):
        self.all_attentions = {}
        # initialize all attentions to random for first epoch
        for loc in self.locations:
            self.all_attentions[loc] = {}
            for mod in self.modalities:
                self.all_attentions[loc][mod] = []
                layers = self.freq_interval_layers[loc][mod]
                for basiclayer in layers:
                    for block in basiclayer.blocks:
                        block_attentions = torch.rand(128, 128, 4, 4, 4) #torch.rand(block_attentions.shape)
                        self.all_attentions[loc][mod].append(block_attentions)
                        
class CrossAttention_v2(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention_v2, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, encoded_mod_features):
        B, C = encoded_mod_features[0].shape
        N = len(encoded_mod_features)
        
        # Step 1: Calculate attention weights using a CrossAttention layer
        attended_tensors = []
        for i in range(N):
            query = encoded_mod_features[i].unsqueeze(0)  # shape: (1, B, C)
            key_value = torch.stack(encoded_mod_features, dim=0)  # shape: (N, B, C)
            attended, _ = self.attention(query, key_value, key_value)  # shape: (1, B, C)
            attended_tensors.append(attended.squeeze(0))
            
        # Step 2: Concatenate attended tensors
        fused_embedding = torch.cat(attended_tensors, dim=1)  # shape: (B, N*C)
        
        return fused_embedding

# Cross attention with modality-specific projections
class CrossAttention_v3(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention_v3, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.mod_projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(nhead)])

    def forward(self, encoded_mod_features):
        B, C = encoded_mod_features[0].shape
        N = len(encoded_mod_features)
        
        attended_tensors = []
        for i in range(N):
            # Project each modality separately
            projected_tensor = self.mod_projections[i](encoded_mod_features[i])
            
            # Calculate attention weights
            query = projected_tensor.unsqueeze(0)  # shape: (1, B, C)
            key_value = torch.stack([self.mod_projections[j](tensor) for j, tensor in enumerate(encoded_mod_features)], dim=0)  # shape: (N, B, C)
            attended, _ = self.attention(query, key_value, key_value)  # shape: (1, B, C)
            attended_tensors.append(attended.squeeze(0))
            
        fused_embedding = torch.cat(attended_tensors, dim=1)  # shape: (B, N*C)
        
        return fused_embedding

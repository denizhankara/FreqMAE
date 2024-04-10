import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

class TSTCC(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone_config = backbone.config
        self.config = args.dataset_config["TSTCC"]

        # components
        self.backbone = backbone
        
        # temporal contrast
        self.temporal_contrast = TC(args)

    def forward(self, aug_freq_input1, aug_freq_input2):
        # get representation
        """
        Input:
            freq_input1: Input of the first augmentation.
            freq_input2: Input of the second augmentation.
        Output:
            mod_features1: Projected mod features of the first augmentation.
            mod_features2: Projected mod features of the second augmentation.
        """
        # compute features
        mod_features1 = self.backbone(aug_freq_input1, class_head=False)
        mod_features2 = self.backbone(aug_freq_input2, class_head=False)
        
        # normalize projection feature vectors
        mod_features1 = F.normalize(mod_features1, dim=1)
        mod_features2 = F.normalize(mod_features2, dim=1)
        
        temp_contrast_loss1, temporal_contrast_features1 = self.temporal_contrast(mod_features1, mod_features2)
        temp_contrast_loss2, temporal_contrast_features2 = self.temporal_contrast(mod_features2, mod_features1)
        
        return [temporal_contrast_features1, temporal_contrast_features2], [temp_contrast_loss1, temp_contrast_loss2]
        
        


###########################################

class TC(nn.Module):
    def __init__(self, args):
        super(TC, self).__init__()
        
        self.args = args
        self.configs = self.args.dataset_config["TSTCC"]
        self.backbone_configs = args.dataset_config[args.model]
        
        
        # self.num_channels = configs.final_out_channels
        self.num_channels = self.backbone_configs["fc_dim"]
        self.timestep = self.args.dataset_config["seq_len"] - 1
        
        
        self.Wk = nn.ModuleList([nn.Linear(self.configs["emb_dim"], self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = self.args.device
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.configs["emb_dim"], self.num_channels // 2),
            nn.BatchNorm1d(self.num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_channels // 2, self.num_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=self.configs["emb_dim"], depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        batch_size = features_aug1.shape[0]
        seq_len = self.args.dataset_config["seq_len"]

        # step 0: split features into subsequence
        split_mod_features1 = features_aug1.reshape(batch_size // seq_len, seq_len, -1)
        split_mod_features1 = split_mod_features1.permute(0, 2, 1)
        split_mod_features2 = features_aug2.reshape(batch_size // seq_len, seq_len, -1)
        split_mod_features2 = split_mod_features2.permute(0, 2, 1)
        
        z_aug1 = split_mod_features1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = split_mod_features2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :] 

        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)

########################################################################################

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, emb_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()


    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t
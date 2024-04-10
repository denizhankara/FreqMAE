import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAE(nn.Module):
    """
    Masked Auto Encoder
    """

    def __init__(self, args, backbone):
        super(MAE, self).__init__()

        self.args = args
        self.config = args.dataset_config["MAE"]
        self.backbone = backbone

    def forward(self, freq_input):
        # compute features
        return self.backbone(freq_input, class_head=False)


def window_masking(
    x: torch.Tensor,
    input_resolution,
    patch_resolution,
    window_size,
    mask_token,
    remove=False,
    mask_len_sparse: bool = False,
    mask_ratio=0.75,
):
    """
    The new masking method, masking the adjacent r*r number of patches together
    Optional whether to remove the mask patch,
    if so, the return value returns one more sparse_restore for restoring the order to x
    Optionally, the returned mask index is sparse length or original length,
    which corresponds to the different size choices of the decoder when restoring the image
    x: [N, L, D]
    r: There are r*r patches in a window
    remove: Whether to remove the mask patch
    mask_len_sparse: Whether the returned mask length is a sparse short length
    """
    B, L, D = x.shape
    h, w = input_resolution[0], input_resolution[1]  # padded image h and w

    # assert L == h * w
    ph, pw = patch_resolution[0], patch_resolution[1]  # num patches h and w
    dh, dw = int(ph // window_size[0]), int(pw // window_size[1])  # window_resolution h and w
    rh, rw = window_size[0], window_size[1]

    noise = torch.rand(B, dw * dh, device=x.device)
    sparse_shuffle = torch.argsort(noise, dim=1)
    sparse_restore = torch.argsort(sparse_shuffle, dim=1)
    sparse_keep = sparse_shuffle[:, : int(dw * dh * (1 - mask_ratio))]  # list of preserved patches

    index_keep_part = torch.div(sparse_keep, dw, rounding_mode="floor") * dw * (rw * rh) + (sparse_keep % dw) * rw
    index_keep = index_keep_part
    for i in range(rh):
        for j in range(rw):
            if i == 0 and j == 0:
                continue
            index_keep = torch.cat([index_keep, index_keep_part + pw * i + j], dim=1)
    index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
    index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
    for i in range(B):
        set_diff = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=False)
        index_mask[i] = set_diff
    index_mask = torch.tensor(index_mask, device=x.device)

    index_shuffle = torch.cat([index_keep, index_mask], dim=1)
    index_restore = torch.argsort(index_shuffle, dim=1)

    if mask_len_sparse:
        mask = torch.ones([B, dw * dh], device=x.device)
        mask[:, : sparse_keep.shape[-1]] = 0
        mask = torch.gather(mask, dim=1, index=sparse_restore)
    else:
        mask = torch.ones([B, L], device=x.device)
        mask[:, : index_keep.shape[-1]] = 0
        mask = torch.gather(mask, dim=1, index=index_restore)

    print(L, mask.sum() / B)

    if remove:
        x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
        # x_masked = rearrange(x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5))
        return x_masked, mask, sparse_restore
    else:
        x_masked = torch.clone(x)
        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = mask_token
        # x_masked = rearrange(x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5))
        return (x_masked, mask)


def find_patch_size(dim):
    for i in range(2, dim):
        if dim % i == 0:
            return i
    return 1


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size_h, grid_size_w = grid_size
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

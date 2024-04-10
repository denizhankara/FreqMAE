import torch
import numpy as np
from einops.einops import rearrange
import random

def generate_mask_input(
    mask_scheme,
    freq_x,
    patch_resolution=(1, 1),
    window_size=(1, 1),
    mask_ratio=0.75,
    spectral_coherence=None,
):
    """
    Optimized window masking: generate patch masks
    """

    B = freq_x.shape[0] # [128, 256, 64] for ACIDS

    ph, pw = patch_resolution[0], patch_resolution[1]  # num patches h and w
    dh, dw = int(ph // window_size[0]), int(pw // window_size[1])  # window_resolution h and w

    rh, rw = window_size[0], window_size[1]
    
    
    if mask_scheme == "harmonictubev2":
        # this can implement the increasing patch sizes for larger frequencies, just handle it during masking
        mask_generator = HarmonicTubeMaskingGenerator1Dv2(input_size = (B, ph, pw), mask_ratio = mask_ratio,device=freq_x.device)
        patch_mask = mask_generator()
        
        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        patch_mask = patch_mask.reshape(B, -1).int().float()
        patch_mask = 1 - patch_mask # invert the mask, zeros for masked patches
        pass
    
    
    if mask_scheme == "harmonictube":
        mask_generator = HarmonicTubeMaskingGenerator1D(input_size = (B, ph, pw), mask_ratio = mask_ratio,device=freq_x.device)
        patch_mask = mask_generator()
        
        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        patch_mask = patch_mask.reshape(B, -1).int().float()
        patch_mask = 1 - patch_mask # invert the mask, zeros for masked patches
        pass
    
    if mask_scheme == "time":
        # time masking, mask consecutive frames wholly to learn temporal correlation
        mask_generator = TimeMaskingGenerator(input_size = (B, ph, pw), mask_ratio = mask_ratio,device=freq_x.device)
        patch_mask = mask_generator()
        
        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        patch_mask = patch_mask.reshape(B, -1).int().float()
        patch_mask = 1 - patch_mask # invert the mask, zeros for masked patches
        pass
    if mask_scheme == "tube":
        # frequency masking, mask consecutive frequencies wholly to learn temporal redundancy
        mask_generator = TubeMaskingGenerator1D(input_size = (B, ph, pw), mask_ratio = mask_ratio,device=freq_x.device)
        patch_mask = mask_generator()
        
        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        patch_mask = patch_mask.reshape(B, -1).int().float()
        patch_mask = 1 - patch_mask # invert the mask, zeros for masked patches
        pass
    if mask_scheme == "time+tube":
        # 50 percent time masking, 50 percent frequency masking
        if random.random() > 0.5:
            mask_generator = TimeMaskingGenerator(input_size = (B, ph, pw), mask_ratio = mask_ratio,device=freq_x.device)
        else:
            mask_generator = TubeMaskingGenerator1D(input_size = (B, ph, pw), mask_ratio = mask_ratio,device=freq_x.device)
        patch_mask = mask_generator()
        
        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        patch_mask = patch_mask.reshape(B, -1).int().float()
        patch_mask = 1 - patch_mask # invert the mask, zeros for masked patches
        pass
        
    if mask_scheme == "coherence":
        # spectral coherence
        freq_x_copy = freq_x.reshape(B, ph, pw, -1)
        patch_mask = generate_patch_mask_coherence(spectral_coherence,freq_x_copy, mask_ratio)
        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        # [b, patch_resolutions]
        patch_mask = patch_mask.reshape(B, -1).int().float()
        # patch_mask = 1 - patch_mask # already inverted for coherence
        
    elif mask_scheme == "correlation":
        # FIXME : This is embedding correlation, change to correlation of actual patches
        patch_mask = generate_mask_input_by_correlation(freq_x, mask_ratio)
        
        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        # [b, patch_resolutions]
        patch_mask = patch_mask.reshape(B, -1).int().float()
        patch_mask = 1 - patch_mask # invert the mask, zeros for masked patches
    
    elif mask_scheme == "similarity":
        # cosine similarity of embeddings
        patch_mask = generate_mask_input_by_similarity(freq_x, mask_ratio) # results in slightly higher masking rate
        
        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        # [b, patch_resolutions]
        patch_mask = patch_mask.reshape(B, -1).int().float()
        patch_mask = 1 - patch_mask # invert the mask, zeros for masked patches

    elif mask_scheme == "varyingresolution":
        # set patch size to a smaller size like [1, 2] for varying resolution
        one_eighth_width = pw // 8
        quarter_width = pw // 4
        half_width = pw // 2

        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        def create_mask_section(B, ph, section_width, resolution, mask_ratio):
            section_width_adjusted = section_width // resolution
            mask_section = torch.zeros([B, ph, section_width_adjusted], device=freq_x.device)
            num_masks = int(section_width_adjusted * mask_ratio)

            batch_indices = torch.arange(B, device=freq_x.device).view(-1, 1, 1)
            height_indices = torch.arange(ph, device=freq_x.device).view(1, -1, 1)

            mask_indices = torch.stack([torch.randperm(section_width_adjusted, device=freq_x.device)[:num_masks] for _ in range(B * ph)]).view(B, ph, -1)
            mask_section[batch_indices, height_indices, mask_indices] = 1

            return mask_section.repeat_interleave(resolution, dim=2)
        
        def create_mask_section_slow(B, ph, section_width, resolution, mask_ratio):
            mask_section = torch.zeros([B, ph, section_width // resolution], device=freq_x.device)
            num_masks = int(mask_section.shape[2] * mask_ratio)
            for b in range(B):
                mask_indices = torch.randperm(mask_section.shape[2], device=freq_x.device)[:num_masks]
                mask_section[b, :, mask_indices] = 1
            return mask_section.repeat_interleave(resolution, dim=2)

        mask_1 = create_mask_section(B, ph, quarter_width, 4, mask_ratio)
        mask_2 = create_mask_section(B, ph, one_eighth_width, 2, mask_ratio)
        mask_3 = create_mask_section(B, ph, one_eighth_width, 1, mask_ratio)
        mask_4 = create_mask_section(B, ph, one_eighth_width, 1, mask_ratio)
        mask_5 = create_mask_section(B, ph, one_eighth_width, 2, mask_ratio)
        mask_6 = create_mask_section(B, ph, quarter_width, 4, mask_ratio)
        
        patch_mask = torch.cat([mask_1, mask_2, mask_3, mask_4, mask_5, mask_6], dim=2)

        # [b, patch_resolutions]
        patch_mask = patch_mask.reshape(B, -1).int().float()
        patch_mask = 1 - patch_mask # invert the mask, zeros for masked patches
    
    elif mask_scheme == "multiresolution":
        # randomized patch size resolution for each item
        
        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        def create_mask_section(B, ph, section_width, resolution, mask_ratio):
            section_width_adjusted = section_width // resolution
            mask_section = torch.zeros([B, ph, section_width_adjusted], device=freq_x.device)
            num_masks = int(section_width_adjusted * mask_ratio)

            batch_indices = torch.arange(B, device=freq_x.device).view(-1, 1, 1)
            height_indices = torch.arange(ph, device=freq_x.device).view(1, -1, 1)

            mask_indices = torch.stack([torch.randperm(section_width_adjusted, device=freq_x.device)[:num_masks] for _ in range(B * ph)]).view(B, ph, -1)
            mask_section[batch_indices, height_indices, mask_indices] = 1

            return mask_section.repeat_interleave(resolution, dim=2)
        
        def create_random_resolution_mask_section(B, ph, pw_adjusted, max_resolution, mask_ratio):
            mask_sections = []
            for resolution in range(1, max_resolution + 1):
                section_width = pw_adjusted * resolution
                mask_section = create_mask_section(B, ph, section_width, resolution, mask_ratio)
                mask_sections.append(mask_section)
            
            mask_sections = torch.cat(mask_sections, dim=2)
            return mask_sections[:, :, :pw]
        
        max_resolution = 4
        pw_adjusted = pw // max_resolution
        
        patch_mask = create_random_resolution_mask_section(B, ph, pw_adjusted, max_resolution, mask_ratio)

        # [b, patch_resolutions]
        patch_mask = patch_mask.reshape(B, -1).int().float()
        patch_mask = 1 - patch_mask # invert the mask, zeros for masked patches

    elif mask_scheme == "multiresolutionv2":
        # change the mask resolution for each batch
        
        # create a random resolution pair (rh,rw) between [1, 1] and window_size pair, that divides ph and pw
        # [1, 1] is the lowest resolution, [window_size[0], window_size[1]] is the highest resolution
        while True:
            rh = torch.randint(1, window_size[0] + 1, (1,)).item()
            rw = torch.randint(1, window_size[1] + 1, (1,)).item()

            if ph % rh == 0 and pw % rw == 0:
                break
            
        dh = ph // rh
        dw = pw // rw
        
        # random mask [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        # [b, patch_resolution_height, window_resolution_width]
        patch_mask = bit_mask.repeat_interleave(rh, dim=1)

        # [b, patch_resolution_height, patch_resolution_width]
        patch_mask = patch_mask.repeat_interleave(rw, dim=2)

        # [b, patch_resolutions]
        patch_mask = patch_mask.reshape(B, -1).int().float()
    
    elif mask_scheme == "uniform":
        # from paper: Uniform Masking: Enabling MAE Pre-training for Pyramid-based Vision Transformers with Locality
        # random mask [b, window_resolution height, window_resolution_width]
    
        # mask_generator = RandomMaskingGenerator(input_size = (ph, pw), mask_ratio = mask_ratio,device=freq_x.device,regular=True)
        # patch_mask = mask_generator()
        
        mask_generator = UniformMaskingGeneratorBatch(input_size = (B, ph, pw), mask_ratio = mask_ratio,device=freq_x.device,regular=True)
        patch_mask = mask_generator()
        # random bit mask, not used anyway [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])
        
        patch_mask = patch_mask.reshape(B, -1).int().float()
        patch_mask = 1 - patch_mask # invert the mask, zeros for masked patches
        pass
    
    elif mask_scheme == "random":
        # random masking, default
        # random mask [b, window_resolution height, window_resolution_width]
        bit_mask = torch.rand([B, dh * dw], device=freq_x.device).argsort(1) >= int(dh * dw * mask_ratio)
        bit_mask = bit_mask.reshape([B, dh, dw])

        # [b, patch_resolution_height, window_resolution_width]
        patch_mask = bit_mask.repeat_interleave(rh, dim=1)

        # [b, patch_resolution_height, patch_resolution_width]
        patch_mask = patch_mask.repeat_interleave(rw, dim=2)

        # [b, patch_resolutions]
        patch_mask = patch_mask.reshape(B, -1).int().float()

    return patch_mask, bit_mask


def mask_input(
    freq_x,
    input_resolution,
    patch_resolution=(1, 1),
    channel_dimension=-1,
    window_size=(1, 1),
    mask_token=None,
    mask_ratio=0.75,
    mask_scheme = "random",
    spectral_coherence=None,
):
    """
    Optimized window masking: get masks and apply to inputs
    In returned values, 1 means masked, 0 means preserved.
    """
    if len(freq_x.shape) > 3:
        """DeepSense [B, c, i, s] -> [B, i * s, c]"""
        b, c = freq_x.shape[0:2]
        x = freq_x.reshape(b, c, -1).permute(0, 2, 1)
    else:
        x = freq_x

    B, L, D = x.shape

    # generate masks
    patch_mask, bit_mask = generate_mask_input(mask_scheme,freq_x, patch_resolution, window_size, mask_ratio,spectral_coherence)

    # [b, patch_resolution, D] or [b, D, patch_resolution]
    channel_repeat = [1, 1, 1]
    channel_repeat[channel_dimension] = D  # (1, 1, D) or (1, D, 1)
    patch_mask_channel = patch_mask.unsqueeze(channel_dimension).tile(channel_repeat)

    # mask_token： [D] -> [1, 1, D]
    mask_token = mask_token.tile([B, 1, 1])

    # masked [b, patch_resolution, 1]
    token_mask = (1 - patch_mask).unsqueeze(-1)

    # mask_token: [b, patch_resolution, 1] @ [b, 1, D] -> [1, patch_resolution, D]
    masked_x = x * patch_mask_channel + token_mask @ mask_token

    return masked_x, 1 - patch_mask.int(), 1 - bit_mask.int()


def pairwise_correlation(X):
    """
    Calculate pairwise correlation between patches in a tensor.
    
    Args:
        X (torch.Tensor): Tensor of shape (B, N, D), where B is batch size, N is the number of patches, and D is the dimension of each patch.
    
    Returns:
        torch.Tensor: Pairwise correlation matrix of shape (B, N, N).
    """
    # Normalize patches
    X_norm = (X - X.mean(dim=-1, keepdim=True)) / X.std(dim=-1, keepdim=True)
    
    # Calculate dot product between normalized patches
    dot_product = torch.bmm(X_norm, X_norm.transpose(1, 2))
    
    # Divide dot product by the number of elements in each patch
    num_elements = X.shape[-1]
    correlation_matrix = dot_product / num_elements

    return correlation_matrix

def generate_mask_input_by_correlation(X, mask_ratio=0.75):
    """
    Generate mask input based on pairwise correlation between patches.
    
    Args:
        X (torch.Tensor): Tensor of shape (B, N, D), where B is batch size, N is the number of patches, and D is the dimension of each patch.
        mask_ratio (float): Ratio of patches to be masked.

    Returns:
        torch.Tensor: Mask input of shape (B, N).
    """
    # Calculate pairwise correlation matrix
    correlation_matrix = pairwise_correlation(X)

    # Get the sorted correlations
    sorted_correlations, _ = torch.sort(correlation_matrix, dim=-1)

    # Calculate the threshold index based on mask ratio
    threshold_idx = int(mask_ratio * sorted_correlations.shape[-1])

    # Select the threshold value for each batch
    threshold_values = sorted_correlations[:, :, threshold_idx].unsqueeze(-1)

    # Calculate the probability of masking a patch based on its correlation with other patches
    mask_prob = (correlation_matrix > threshold_values).float().mean(dim=-1)

    # Generate the mask input using the calculated probabilities
    mask_input = torch.bernoulli(1 - mask_prob)

    return mask_input


def calculate_patch_embedding_similarity(patch_embeddings):
    """
    Calculate cosine similarity between patch embeddings.
    
    Args:
        patch_embeddings (torch.Tensor): Tensor of shape (B, N, D), where B is batch size, N is the number of patches, and D is the dimension of each patch.

    Returns:
        torch.Tensor: Cosine similarity matrix of shape (B, N, N).
    """
    # Normalize patch embeddings
    normalized_embeddings = torch.nn.functional.normalize(patch_embeddings, dim=-1)
    
    # Calculate cosine similarity matrix
    similarity_matrix = torch.bmm(normalized_embeddings, normalized_embeddings.transpose(1, 2))
    
    return similarity_matrix


def generate_mask_input_by_similarity(X, mask_ratio=0.75):
    """
    Generate mask input based on cosine similarity between patches.
    
    Args:
        X (torch.Tensor): Tensor of shape (B, N, D), where B is batch size, N is the number of patches, and D is the dimension of each patch.
        mask_ratio (float): Ratio of patches to be masked.

    Returns:
        torch.Tensor: Mask input of shape (B, N).
    """
    # Calculate cosine similarity matrix
    similarity_matrix = calculate_patch_embedding_similarity(X)

    # Get the sorted similarities
    sorted_similarities, _ = torch.sort(similarity_matrix, dim=-1)

    # Calculate the threshold index based on mask ratio
    threshold_idx = int(mask_ratio * sorted_similarities.shape[-1])

    # Select the threshold value for each batch
    threshold_values = sorted_similarities[:, :, threshold_idx].unsqueeze(-1)

    # Calculate the boolean mask for similarities higher than the threshold
    high_similarity_mask = (similarity_matrix > threshold_values).float()

    # For each patch, find the patch with the highest similarity
    max_similarity_idx = torch.argmax(similarity_matrix, dim=-1)

    # Create a mask with the shape of the similarity matrix
    B, N, _ = similarity_matrix.shape
    idx_range = torch.arange(N, device=similarity_matrix.device)
    idx_mask = idx_range.view(1, -1).expand(B, -1)

    # Create a boolean mask to indicate the highest similarity patches
    highest_similarity_mask = (idx_mask == max_similarity_idx).float().unsqueeze(-1)

    # Calculate the probability of masking a patch based on its similarity with other patches
    mask_prob = (high_similarity_mask * highest_similarity_mask).mean(dim=-1)

    # Generate the mask input using the calculated probabilities
    mask_input = torch.bernoulli(1 - mask_prob)

    return mask_input

## spectral coherence
def cross_coherence(input_tensor, window_size=11):
    # Input tensor shape: (batch, channel, time, freq)
    batch, channel, time, freq = input_tensor.shape
    
    # Compute the cross-spectral density between successive time frames
    cross_spectral_density = input_tensor[:, :, :-1, :] * input_tensor[:, :, 1:, :].conj()
    # cross_spectral_density shape: (batch, channel, time - 1, freq)
    
    # Compute the auto-spectral density for each time frame
    auto_spectral_density = input_tensor * input_tensor.conj()
    
    # Smooth the cross-spectral density and auto-spectral density tensors
    window = torch.hann_window(window_size, dtype=torch.float).view(1, 1, -1).to(input_tensor.device)
    smoothed_cross_spectral_density = torch.nn.functional.conv1d(
        cross_spectral_density.view(batch * channel, 1, (time - 1) * freq),
        window,
        padding=(window_size - 1) // 2) / window_size
    smoothed_auto_spectral_density = torch.nn.functional.conv1d(
        auto_spectral_density.view(batch * channel, 1, time * freq),
        window,
        padding=(window_size - 1) // 2) / window_size
    
    # Reshape the tensors
    smoothed_cross_spectral_density = smoothed_cross_spectral_density.view(batch, channel, time - 1, freq)
    smoothed_auto_spectral_density = smoothed_auto_spectral_density.view(batch, channel, time, freq)
    
    # Compute the cross-coherence
    cross_coherence = torch.abs(smoothed_cross_spectral_density) / torch.sqrt(
        smoothed_auto_spectral_density[:, :, :-1, :] * smoothed_auto_spectral_density[:, :, 1:, :])
    
    return cross_coherence

def generate_patch_mask_coherence(coherence, embeddings,ph,pw, mask_ratio=0.75):
    # Compute the average cross-coherence across channels and frequencies
    avg_coherence = coherence.mean(dim=(1, 3))  # Shape: (batch, time - 1)
    
    # Normalize the average cross-coherence to the range [0, 1]
    normalized_coherence = (avg_coherence - avg_coherence.min()) / (avg_coherence.max() - avg_coherence.min())
    
    # Calculate the threshold value for the desired mask ratio
    threshold = torch.quantile(normalized_coherence, 1 - mask_ratio)
    
    # Calculate the probabilities of masking each patch
    mask_prob = torch.where(normalized_coherence > threshold, 1 - mask_ratio, mask_ratio)
    
    # Draw samples from Bernoulli distributions with these probabilities
    time_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool)
    
    # Generate the final patch mask
    patch_mask = torch.zeros(embeddings.shape[:-1], dtype=torch.bool)
    for i in range(1, embeddings.shape[1]):
        patch_mask[:, i, :] = time_mask[:, i - 1].unsqueeze(-1)
    
    # to device
    patch_mask = patch_mask.to(embeddings.device)
    return patch_mask


class TubeMaskingGenerator2D: # for video 2D frame or stft frames
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask


class TubeMaskingGenerator1D:
    def __init__(self, input_size, mask_ratio, device):
        self.batch_size, self.frames, self.width = input_size
        self.num_patches_per_frame = self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        self.device = device

    def __repr__(self):
        repr_str = "Masks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        half_width = (self.width + 1) // 2
        half_num_masks_per_frame = self.num_masks_per_frame // 2

        mask_per_frame_half = torch.cat([
            torch.zeros(half_width - half_num_masks_per_frame, device=self.device),
            torch.ones(half_num_masks_per_frame, device=self.device),
        ])

        mask_per_frame_half_batch = mask_per_frame_half.repeat(self.batch_size, 1)
        mask_per_frame_half_batch = mask_per_frame_half_batch.view(self.batch_size, -1)

        mask_per_frame_half_batch_shuffled = torch.zeros_like(mask_per_frame_half_batch)
        for i in range(self.batch_size):
            mask_per_frame_half_batch_shuffled[i] = mask_per_frame_half_batch[i, torch.randperm(half_width, device=self.device)]

        mask_per_frame_batch = torch.cat([mask_per_frame_half_batch_shuffled, mask_per_frame_half_batch_shuffled.flip(dims=[1])[:, :self.width // 2]], dim=1)

        mask = mask_per_frame_batch.repeat(1, self.frames).view(self.batch_size, self.frames, self.width)
        return mask


class HarmonicTubeMaskingGenerator1D:
    def __init__(self, input_size, mask_ratio, device):
        self.batch_size, self.frames, self.width = input_size
        self.num_patches_per_frame = self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        self.device = device

    def __repr__(self):
        repr_str = "Masks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        half_width = (self.width + 1) // 2
        half_num_masks_per_frame = self.num_masks_per_frame // 2

        mask_per_frame_half = torch.cat([
            torch.zeros(half_width - half_num_masks_per_frame, device=self.device),
            torch.ones(half_num_masks_per_frame, device=self.device),
        ])

        mask_per_frame_half_batch = mask_per_frame_half.repeat(self.batch_size, 1)
        mask_per_frame_half_batch = mask_per_frame_half_batch.view(self.batch_size, -1)

        mask_per_frame_half_batch_shuffled = torch.zeros_like(mask_per_frame_half_batch)
        for i in range(self.batch_size):
            mask_per_frame_half_batch_shuffled[i] = mask_per_frame_half_batch[i, torch.randperm(half_width, device=self.device)]

        mask_per_frame_batch = torch.cat([mask_per_frame_half_batch_shuffled, mask_per_frame_half_batch_shuffled.flip(dims=[1])[:, :self.width // 2]], dim=1)

        # Extend the mask to (2*i+half_width), (3*i+half_width), and so on
        for i in range(self.batch_size):
            mask = mask_per_frame_batch[i, half_width:]
            for j in range(half_width, self.width):
                if mask[j - half_width] == 1:
                    multiplier = 2
                    while (multiplier * j) - half_width < self.width:
                        mask_per_frame_batch[i, (multiplier * j) - half_width] = 1
                        multiplier += 1

        mask = mask_per_frame_batch.repeat(1, self.frames).view(self.batch_size, self.frames, self.width)
        return mask

class HarmonicTubeMaskingGenerator1Dv2:
    # skip masking one of the indices (k*i + half_width) and print the value of k,
    def __init__(self, input_size, mask_ratio, device):
        self.batch_size, self.frames, self.width = input_size
        self.num_patches_per_frame = self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        self.device = device

    def __repr__(self):
        repr_str = "Masks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        half_width = (self.width + 1) // 2
        half_num_masks_per_frame = self.num_masks_per_frame // 2

        mask_per_frame_half = torch.cat([
            torch.zeros(half_width - half_num_masks_per_frame, device=self.device),
            torch.ones(half_num_masks_per_frame, device=self.device),
        ])

        mask_per_frame_half = mask_per_frame_half[torch.randperm(half_width, device=self.device)]

        # Enforce harmonic structure
        indices_to_skip = torch.arange(1, half_width // 2, device=self.device)

        if len(indices_to_skip) > 0:
            k_idx = torch.randint(len(indices_to_skip), (1,)).item()
            k = indices_to_skip[k_idx].item()
            print("k:", k)
        else:
            k = None

        for i in range(half_width, self.width, 2):
            if mask_per_frame_half[i - half_width] == 1:
                j = 2
                while i * j < self.width:
                    if j != k:
                        mask_per_frame_half[i * j - half_width] = 1
                    j += 1

        mask_per_frame = torch.cat([mask_per_frame_half, mask_per_frame_half.flip(0)])

        mask_per_frame_batch = mask_per_frame.repeat(self.batch_size, 1)
        mask_per_frame_batch = mask_per_frame_batch.view(self.batch_size, -1)

        mask = mask_per_frame_batch.repeat(1, self.frames).view(self.batch_size, self.frames, self.width)
        return mask

class UniformMaskingGenerator:
    def __init__(self, input_size, mask_ratio, regular=False, device=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.regular = regular

        if regular:
            assert mask_ratio == 0.75
        
            candidate_list = []
            while True: # add more
                for j in range(4):
                    candidate = np.ones(4)
                    candidate[j] = 0
                    candidate_list.append(candidate)
                if len(candidate_list) * 4 >= self.num_patches * 2:
                    break
            self.mask_candidate = np.vstack(candidate_list) 
            print('using regular, mask_candidate shape = ', 
                  self.mask_candidate.shape)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, regular {}".format(
            self.num_patches, self.num_mask, self.regular
        )
        return repr_str

    def __call__(self):
        if not self.regular:
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
        else:
            mask = self.mask_candidate.copy()
            np.random.shuffle(mask)
            mask = rearrange(mask[:self.num_patches//4], '(h w) (p1 p2) -> (h p1) (w p2)', 
                             h=self.height//2, w=self.width//2, p1=2, p2=2)
            mask = mask.flatten()

        return mask 

class UniformMaskingGeneratorBatch:
    def __init__(self, input_size, mask_ratio, regular=False, device=None):
        self.device = device if device else torch.device('cpu')

        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3

        self.batch_size, self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.regular = regular

        if regular:
            assert mask_ratio == 0.75
        
            candidate_list = []
            while True:
                for j in range(4):
                    candidate = torch.ones(4, device=self.device)
                    candidate[j] = 0
                    candidate_list.append(candidate)
                if len(candidate_list) * 4 >= self.num_patches * 2:
                    break
            self.mask_candidate = torch.stack(candidate_list)
            # print('using regular, mask_candidate shape = ', 
                #   self.mask_candidate.shape)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, regular {}".format(
            self.num_patches, self.num_mask, self.regular
        )
        return repr_str

    def __call__(self):
        B = self.batch_size
        mask_list = []
        for _ in range(B):
            if not self.regular:
                mask = torch.cat([
                    torch.zeros(self.num_patches - self.num_mask, device=self.device),
                    torch.ones(self.num_mask, device=self.device),
                ])
                mask = mask[torch.randperm(mask.size(0))]
            else:
                mask = self.mask_candidate.clone()
                # np.random.shuffle(mask)
                mask = mask[torch.randperm(mask.size(0))]
                mask = rearrange(mask[:self.num_patches//4], '(h w) (p1 p2) -> (h p1) (w p2)', 
                                h=self.height//2, w=self.width//2, p1=2, p2=2)
                # mask = mask.flatten()
            mask_list.append(mask)
        mask = torch.stack(mask_list, 0)
        return mask
    
class TimeMaskingGenerator:
    def __init__(self, input_size, mask_ratio, device):
        self.batch_size, self.frames, self.width = input_size
        self.num_patches_per_frame = self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masked_frames = int(mask_ratio * self.frames)
        self.device = device

    def __repr__(self):
        repr_str = "Masks: total patches {}, mask patches {}".format(
            self.total_patches, self.num_masked_frames * self.num_patches_per_frame
        )
        return repr_str

    def __call__(self):
        # Create a binary mask for frames: 1 indicates a frame to mask, 0 indicates a frame to keep
        frame_mask = torch.cat([
            torch.ones(self.num_masked_frames, device=self.device),
            torch.zeros(self.frames - self.num_masked_frames, device=self.device),
        ])
        frame_mask = frame_mask[torch.randperm(self.frames, device=self.device)]  # Shuffle the frame mask

        # Extend the frame mask to the width of each frame
        mask = frame_mask.view(self.frames, 1).repeat(1, self.width)

        # Extend the mask to all items in the batch
        mask = mask.repeat(self.batch_size, 1, 1)

        return mask


def selectMaskScheme(options, probabilities):
    """
    Select a mask scheme based on the given probabilities.
    
    Parameters:
    - options: A list of possible mask schemes.
    - probabilities: A list of probabilities associated with each mask scheme.

    Returns:
    A randomly selected mask scheme.
    """
    return np.random.choice(options, p=probabilities)

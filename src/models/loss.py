import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.utilities import reduce
from general_utils.tensor_utils import extract_non_diagonal_matrix
from models.CMCV2Modules import split_features


class DINOLoss(nn.Module):
    """The loss function.
    We subclass the `nn.Module` becuase we want to create a buffer for the
    logits center of the teacher.
    Parameters
    ----------
    out_dim : int
        The dimensionality of the final layer (we computed the softmax over).
    teacher_temp, student_temp : float
        Softmax temperature of the teacher resp. student.
    center_momentum : float
        Hyperparameter for the exponential moving average that determines
        the center logits. The higher the more the running average matters.
    """

    def __init__(self, args):
        super().__init__()
        self.config = args.dataset_config[args.learn_framework]

        # hyperparameters
        self.temperature_s = self.config["temperature_student"]
        self.temperature_t = self.config["temperature_teacher"]
        self.center_momentum = self.config["center_momentum"]

        # centering vector
        self.register_buffer("center", torch.zeros(1, self.config["emb_dim"]))

    def forward(self, student_output, teacher_output, idx=None):
        """Evaluate loss.
        Parameters
        ----------
        student_output, teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` representing
            logits. The length is equal to number of crops.
            Note that student processed all crops and that the two initial crops
            are the global ones.
        Returns
        -------
        loss : torch.Tensor
            Scalar representing the average loss.
        """
        student_out = student_output / self.temperature_s

        # teacher centering and sharpening
        teacher_out = F.softmax((teacher_output - self.center) / self.temperature_t, dim=-1)
        teacher_out = teacher_out.detach()

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    def H(self, feature_s, feature_t):
        """Sub-forward function used to calculate the loss

        Args:
            z_i (_type_): student predicted feature
            z_j (_type_): teacher predicted feature
        """
        feature_t = feature_t.detach()
        loss_s = F.softmax(feature_s / self.temperature_s, dim=1)
        loss_t = F.softmax((feature_t - self.center) / self.temperature_t, dim=1)
        return -(loss_t * torch.log(loss_s)).sum(dim=1).mean()

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.
        Compute the exponential moving average.
        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        # bc = cat([t1, t2]).mean(dim=0)
        batch_center = torch.cat((teacher_output[0], teacher_output[1])).mean(dim=0, keepdim=True)  # (1, out_dim)
        # C = (m * c) + (1-m) * (bc)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class SimCLRLoss(nn.Module):
    def __init__(self, args):
        super(SimCLRLoss, self).__init__()
        self.temperature = args.dataset_config[args.learn_framework]["temperature"]
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.args = args

    def mask_correlated_samples(self, batch_size):
        """
        Mark diagonal elements and elements of upper-right and lower-left diagonals as 0.
        """
        N = 2 * batch_size
        diag_mat = torch.eye(batch_size).to(self.args.device)
        mask = torch.ones((N, N)).to(self.args.device)

        mask = mask.fill_diagonal_(0)
        mask[0:batch_size, batch_size : 2 * batch_size] -= diag_mat
        mask[batch_size : 2 * batch_size, 0:batch_size] -= diag_mat

        mask = mask.bool()

        return mask

    def forward(self, z_i, z_j, idx=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        # Calculate similarity
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / N

        return loss


class MoCoLoss(nn.Module):
    """MoCo Loss function
    https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
    """

    def __init__(self, args):
        super(MoCoLoss, self).__init__()
        self.args = args
        self.T = args.dataset_config["MoCo"]["temperature"]

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        # Einstein sum is more intuitive
        logits = torch.einsum("nc,mc->nm", [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long)).to(self.args.device)

        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, q, k, idx=None):
        q1, q2 = q
        k1, k2 = k
        loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        return loss


class CMCLoss(nn.Module):
    def __init__(self, args):
        super(CMCLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["CMC"]
        self.batch_size = args.batch_size
        self.temperature = args.dataset_config[args.learn_framework]["temperature"]
        self.modalities = args.dataset_config["modality_names"]

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        """
        Mark diagonal elements and elements of upper-right and lower-left diagonals as 0.
        """
        N = 2 * batch_size
        diag_mat = torch.eye(batch_size).to(self.args.device)
        mask = torch.ones((N, N)).to(self.args.device)

        mask = mask.fill_diagonal_(0)
        mask[0:batch_size, batch_size : 2 * batch_size] -= diag_mat
        mask[batch_size : 2 * batch_size, 0:batch_size] -= diag_mat

        mask = mask.bool()

        return mask

    def forward_similiarity(self, z_i, z_j, idx=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        # Calculate similarity
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim = sim.reshape(sim.shape[0], sim.shape[1])
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / N

        return loss

    def forward(self, mod_features, index):
        """
        Iterate over pairs of modalities and calculate the loss.
        """
        loss = 0

        for i, mod1 in enumerate(self.modalities):
            for mod2 in self.modalities[i + 1 :]:
                loss += self.forward_similiarity(mod_features[mod1], mod_features[mod2])
        return loss


class CMCV2Loss(nn.Module):
    def __init__(self, args):
        super(CMCV2Loss, self).__init__()
        self.args = args
        self.config = args.dataset_config["CMCV2"]
        self.modalities = args.dataset_config["modality_names"]
        self.temperature = args.dataset_config[args.learn_framework]["temperature"]
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.orthonal_loss_f = nn.CosineEmbeddingLoss(reduction="mean")

    def mask_correlated_samples(self, batch_size):
        """
        Mark diagonal elements and elements of upper-right and lower-left diagonals as 0.
        """
        N = 2 * batch_size
        diag_mat = torch.eye(batch_size).to(self.args.device)
        mask = torch.ones((N, N)).to(self.args.device)

        mask = mask.fill_diagonal_(0)
        mask[0:batch_size, batch_size : 2 * batch_size] -= diag_mat
        mask[batch_size : 2 * batch_size, 0:batch_size] -= diag_mat

        mask = mask.bool()

        return mask

    def forward_contrastive_loss(self, mod1_feature, mod2_feature):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        """
        batch_size = mod1_feature.shape[0]
        N = 2 * batch_size

        # Calculate similarity
        z = torch.cat((mod1_feature, mod2_feature), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim = sim.reshape(sim.shape[0], sim.shape[1])
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)

        # Compute loss
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / N

        return loss

    def forward_orthogonality_loss(self, embeddings1, embeddings2):
        """
        Compute the orthogonality loss for the modality features. No cross-sample operation is involved.
        input shape: [b. dim]
        """
        batch = embeddings1.shape[0]
        orthogonal_loss = self.orthonal_loss_f(
            embeddings1,
            embeddings2,
            target=torch.ones(batch).to(embeddings1.device),
        )

        return orthogonal_loss

    def forward(self, mod_features1, mod_features2, index=None):
        """
        loss = shared contrastive loss + mod contrastive loss + orthogonality loss
        """
        # split features into "shared" space and "private" space
        split_mod_features1, split_mod_features2 = {}, {}
        for mod in self.modalities:
            dim = mod_features1[mod].shape[1]
            split_mod_features1[mod] = {
                "shared": mod_features1[mod][:, 0 : (dim // 2)],
                "private": mod_features1[mod][:, (dim // 2) :],
            }

            dim = mod_features2[mod].shape[1]
            split_mod_features2[mod] = {
                "shared": mod_features2[mod][:, 0 : (dim // 2)],
                "private": mod_features2[mod][:, (dim // 2) :],
            }

        # shared space contrastive loss
        shared_contrastive_loss = 0
        for split_mod_features in [split_mod_features1, split_mod_features2]:
            for i, mod1 in enumerate(self.modalities):
                for mod2 in self.modalities[i + 1 :]:
                    shared_contrastive_loss += self.forward_contrastive_loss(
                        split_mod_features[mod1]["shared"],
                        split_mod_features[mod2]["shared"],
                    )

        # private space contrastive loss
        private_contrastive_loss = 0
        for mod in self.modalities:
            private_contrastive_loss += self.forward_contrastive_loss(
                split_mod_features1[mod]["private"],
                split_mod_features2[mod]["private"],
            )

        # orthogonality loss
        orthogonality_loss = 0
        for split_mod_features in [split_mod_features1, split_mod_features2]:
            for i, mod in enumerate(self.modalities):
                # orthognoality between shared and private space
                orthogonality_loss += self.forward_orthogonality_loss(
                    split_mod_features[mod]["shared"],
                    split_mod_features[mod]["private"],
                )

                # orthogonality between modalities
                for mod2 in self.modalities[i + 1 :]:
                    orthogonality_loss += self.forward_orthogonality_loss(
                        split_mod_features[mod]["private"],
                        split_mod_features[mod2]["private"],
                    )

        loss = shared_contrastive_loss + private_contrastive_loss + orthogonality_loss

        return loss


class CMCV3Loss(nn.Module):
    def __init__(self, args):
        super(CMCV3Loss, self).__init__()
        self.args = args
        self.config = args.dataset_config["CMCV2"]
        self.modalities = args.dataset_config["modality_names"]
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=-1)
        self.orthonal_loss_f = nn.CosineEmbeddingLoss(reduction="mean")
        self.intra_ranking_loss_f = nn.MarginRankingLoss(margin=self.config["intra_rank_margin"], reduction="mean")
        self.inter_ranking_loss_f = nn.MarginRankingLoss(margin=self.config["inter_rank_margin"], reduction="mean")

        # decide the temperature
        if isinstance(self.config["temperature"], dict):
            self.temperature = self.config["temperature"][args.model]
        else:
            self.temperature = self.config["temperature"]

    def mask_correlated_samples(self, seq_len, batch_size, temporal=False):
        """
        Return a mask where the positive sample locations are 0, negative sample locations are 1.
        """
        if temporal:
            """Extract comparison between sequences, output: [B * Seq, B * Seq]"""
            mask = torch.ones([batch_size, batch_size], dtype=bool).to(self.args.device)
            mask = mask.fill_diagonal_(0)
            mask = mask.repeat_interleave(seq_len, dim=0).repeat_interleave(seq_len, dim=1)
        else:
            """Extract [2N, 2N-2] negative locations from [2N, 2N] matrix, output: [seq, B, B]"""
            N = 2 * batch_size
            diag_mat = torch.eye(batch_size).to(self.args.device)
            mask = torch.ones((N, N)).to(self.args.device)

            mask = mask.fill_diagonal_(0)
            mask[0:batch_size, batch_size : 2 * batch_size] -= diag_mat
            mask[batch_size : 2 * batch_size, 0:batch_size] -= diag_mat

            mask = mask.unsqueeze(0).repeat(seq_len, 1, 1).bool()

        return mask

    def forward_contrastive_loss(self, embeddings1, embeddings2, finegrain=False):
        """
        Among sequences, only samples at paired temporal locations are compared.
        embeddings shape: [b, seq, dim]
        """
        # get shape
        batch, seq, dim = embeddings1.shape

        # Put the compared dimension into the second dimension
        if finegrain:
            """Compare within the sequences, [b, seq, dim]"""
            in_embeddings1 = embeddings1
            in_embeddings2 = embeddings2
            N = 2 * seq
            dim_parallel = batch
            dim_compare = seq
        else:
            """Compare between the sequences, [seq, b, dim]"""
            in_embeddings1 = embeddings1.transpose(0, 1)
            in_embeddings2 = embeddings2.transpose(0, 1)
            N = 2 * batch
            dim_parallel = seq
            dim_compare = batch

        # Calculate similarity
        z = torch.cat((in_embeddings1, in_embeddings2), dim=1)
        sim = self.similarity_f(z.unsqueeze(2), z.unsqueeze(1)) / self.temperature
        sim_i_j = torch.diagonal(sim, dim_compare, dim1=-2, dim2=-1)
        sim_j_i = torch.diagonal(sim, -dim_compare, dim1=-2, dim2=-1)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=1).reshape(dim_parallel, N, 1)
        negative_samples = sim[self.mask_correlated_samples(dim_parallel, dim_compare)].reshape(dim_parallel, N, -1)

        # Compute loss
        labels = torch.zeros(dim_parallel * N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=2).reshape(dim_parallel * N, -1)
        contrastive_loss = self.criterion(logits, labels)

        return contrastive_loss

    def forward_orthogonality_loss(self, embeddings1, embeddings2):
        """
        Compute the orthogonality loss for the modality features. No cross-sample operation is involved.
        input shape: [b, seq_len, dim]
        We use y=-1 to make embedding1 and embedding2 orthogonal.
        """
        # convert [batch, seq, dim] to [batch * seq, dim]
        flat_embeddings1 = embeddings1.reshape(-1, embeddings1.shape[-1])
        flat_embeddings2 = embeddings2.reshape(-1, embeddings2.shape[-1])

        batch = flat_embeddings1.shape[0]
        orthogonal_loss = self.orthonal_loss_f(
            flat_embeddings1,
            flat_embeddings2,
            target=-torch.ones(batch).to(embeddings1.device),
        )

        return orthogonal_loss

    def forward_temporal_intra_ranking_loss(self, embeddings):
        """
        Enforce the temporal correlations within each subsequence, for both shared space and private space.
        sim(anchor, sample1) >= sim(anchor, sample2) ... >= sim(anchor, sampleN)
        Input: the sample embeddings, [b, seq, dim], where the sequence samples are temporally correlated.
        """
        # [batch, seq, seq]
        sim = self.similarity_f(embeddings.unsqueeze(2), embeddings.unsqueeze(1))
        B, N = sim.shape[:2]

        # Step 1: Create diagonal masks [b, seq, seq - 1]
        base_mask = torch.zeros(N, N - 1)

        # upper right are ones
        upper_right = torch.triu(torch.ones(N, N - 1), diagonal=1)

        # bottom left are negative ones
        bottom_left = torch.rot90(upper_right, k=2)

        # sum together
        mask = base_mask + upper_right + -1 * bottom_left

        # Step 2: Flatten the inputs and targets， left_half - right_half >= 0
        sim_left_half = sim[:, :, :-1].reshape(B, -1)
        sim_right_half = sim[:, :, 1:].reshape(B, -1)

        # mask [seq, seq - 1] -> [B, seq, seq - 1] -> [B, seq ^ 2]
        y = mask.unsqueeze(0).repeat(B, 1, 1).reshape(B, -1).to(self.args.device)

        # Step 3: Loss, (Step 1 is moved to Step 4 (sim[:, :, -1] - sim[:, :, 1:] == x1 - x2 in MarginRankingLoss))
        # target requires same number of dimension as the inputs
        # Masked portions are zero values and do not contribute to the loss calculation
        correlation_loss = self.intra_ranking_loss_f(sim_left_half, sim_right_half, y)

        return correlation_loss

    def forward_temporal_inter_ranking_loss(self, embeddings):
        """
        Enforce temporally close samples to be closer than temporally distant samples.
        embeddings: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = embeddings.shape
        in_embeddings = embeddings.reshape(batch_size * seq_len, dim)

        # calculate the Euclidean distance between any pairs of samples, [batch * seq, batch * seq]
        distance = torch.cdist(in_embeddings, in_embeddings, p=2)
        distance = distance.reshape(batch_size, seq_len, batch_size, seq_len)
        distance = distance.permute(0, 2, 1, 3)  # [b, b, seq, seq]

        # sequence level distance, [batch, batch]
        mask = torch.ones(batch_size * seq_len, batch_size * seq_len).to(self.args.device).fill_diagonal_(0)
        mask = mask.reshape(batch_size, seq_len, batch_size, seq_len).permute(0, 2, 1, 3)  # [b, b, seq, seq]
        distance = (distance * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3])

        # in-sequence distance should be lower than inter-sequence distance
        avg_intra_seq_dist = torch.diagonal(distance, 0, dim1=0, dim2=1).repeat_interleave(batch_size - 1)
        avg_inter_seq_dist = extract_non_diagonal_matrix(distance).flatten()

        # target: x1 < x2, therefore y = -1
        ranking_loss = self.inter_ranking_loss_f(
            avg_intra_seq_dist,
            avg_inter_seq_dist,
            -torch.ones_like(avg_intra_seq_dist).to(self.args.device),
        )

        return ranking_loss

    def forward_coarse_temporal_contrastive_loss(self, embeddings):
        """
        In the coarse-grained temporal contrastive loss, samples within a sequence are considered positive pairs, and
        samples from different sequences are considered negative pairs.
        """
        # [b, seq, dim]
        mod_seq_features = embeddings
        batch_size, seq_len, dim = mod_seq_features.shape
        mod_seq_features = mod_seq_features.reshape(batch_size * seq_len, dim)

        # Calculate similarity, since similarity is computed within the sequence
        z = mod_seq_features
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # extract positive samples
        # 1. Extract diagnoal of batch * batch
        sim_i_i = sim.reshape(batch_size, seq_len, batch_size, seq_len)  # extract batch diagonal
        sim_i_i = torch.diagonal(sim_i_i, 0, dim1=0, dim2=2).permute(2, 0, 1)  # [b, seq, seq]

        # 2. Extract all off-diagonal elements for each sequence, [b * seq, seq - 1]
        positive_samples = extract_non_diagonal_matrix(sim_i_i).reshape([batch_size * seq_len, seq_len - 1])

        # extract negative samples
        sim_neg_mask = self.mask_correlated_samples(seq_len, batch_size, temporal=True)
        negative_samples = sim[sim_neg_mask].reshape(batch_size * seq_len, -1)

        # Compute loss
        labels = torch.zeros(batch_size * seq_len).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        contrastive_loss = self.criterion(logits, labels)

        return contrastive_loss

    def forward(self, mod_features1, mod_features2, index=None):
        """
        loss = shared contrastive loss + private contrastive loss + orthogonality loss + temporal correlation loss
        Procedure:
            (1) Split the features into (batch, subsequence, shared/private).
            (2) For each batch, compute the shared contrastive loss between modalities.
            (3) For each batch and modality, compute the private contrastive loss between samples.
            (4) Compute orthogonality loss beween shared-private and private-private representations.
            (5) For each subsequence, compute the temporal correlation loss.
        """
        seq_len = self.args.dataset_config["seq_len"]

        # Step 0: Reshape features
        reshaped_mod_features1, reshaped_mod_features2 = {}, {}
        for mod in self.modalities:
            reshaped_mod_features1[mod] = mod_features1[mod].reshape(-1, seq_len, mod_features1[mod].shape[-1])
            reshaped_mod_features2[mod] = mod_features2[mod].reshape(-1, seq_len, mod_features2[mod].shape[-1])

        # Step 1: split features into "shared" space and "private" space of each (mod, subsequence), [b, seq, dim]
        split_mod_features1 = split_features(reshaped_mod_features1)
        split_mod_features2 = split_features(reshaped_mod_features2)

        # Step 2: shared space contrastive loss
        shared_contrastive_loss = 0
        if self.args.tag == "noPrivate":
            for mod_features in [reshaped_mod_features1, reshaped_mod_features2]:
                for i, mod1 in enumerate(self.modalities):
                    for mod2 in self.modalities[i + 1 :]:
                        shared_contrastive_loss += self.forward_contrastive_loss(
                            mod_features[mod1],
                            mod_features[mod2],
                        )
        else:
            for split_mod_features in [split_mod_features1, split_mod_features2]:
                for i, mod1 in enumerate(self.modalities):
                    for mod2 in self.modalities[i + 1 :]:
                        shared_contrastive_loss += self.forward_contrastive_loss(
                            split_mod_features[mod1]["shared"],
                            split_mod_features[mod2]["shared"],
                        )

        # Step 3: private space contrastive loss
        private_contrastive_loss = 0
        for mod in self.modalities:
            private_contrastive_loss += self.forward_contrastive_loss(
                split_mod_features1[mod]["private"],
                split_mod_features2[mod]["private"],
            )

        # Step 4: temporal consistency loss
        temporal_consistency_loss = 0
        for mod_features in [reshaped_mod_features1, reshaped_mod_features2]:
            for mod in self.modalities:
                if self.args.tag == "wIntraRank":
                    temporal_consistency_loss += self.forward_temporal_intra_ranking_loss(mod_features[mod])
                elif self.args.tag == "wTempCon":
                    temporal_consistency_loss += self.forward_coarse_temporal_contrastive_loss(mod_features[mod])
                else:
                    temporal_consistency_loss += self.forward_temporal_inter_ranking_loss(mod_features[mod])

        # Step 5: orthogonality loss
        orthogonality_loss = 0
        for split_mod_features in [split_mod_features1, split_mod_features2]:
            for i, mod in enumerate(self.modalities):
                # orthognoality between shared, private, and temporal space
                orthogonality_loss += self.forward_orthogonality_loss(
                    split_mod_features[mod]["shared"],
                    split_mod_features[mod]["private"],
                )

                # orthogonality between modalities
                for mod2 in self.modalities[i + 1 :]:
                    orthogonality_loss += self.forward_orthogonality_loss(
                        split_mod_features[mod]["private"],
                        split_mod_features[mod2]["private"],
                    )

        # print(
        #     f"shared: {shared_contrastive_loss: .3f}, private: {private_contrastive_loss: .3f}, ",
        #     f"orthogonality: {orthogonality_loss: .3f}, temporal: {temporal_consistency_loss: .3f}",
        # )

        loss = (
            shared_contrastive_loss * self.config["shared_contrastive_loss_weight"]
            + private_contrastive_loss * self.config["private_contrastive_loss_weight"]
            + orthogonality_loss * self.config["orthogonal_loss_weight"]
            + temporal_consistency_loss * self.config["rank_loss_weight"]
        )
        return loss


class CosmoLoss(nn.Module):
    def __init__(self, args):
        """Based on SupConLoss from "Supervised Contrastive Loss" paper."""
        super(CosmoLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["Cosmo"]
        self.temperature = self.config["temperature"]
        self.contrast_mode = self.config["contrast_mode"]
        self.base_temperature = self.config["temperature"]

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, feat_dim].
        Returns:
            A loss scalar.
        """
        device = self.args.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # change to [n_views*bsz, dim]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, dim=1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits, z_i * z_a / T
        similarity_matrix = torch.matmul(anchor_feature, contrast_feature.T)
        similarity_matrix = torch.div(similarity_matrix, self.temperature)

        # tile mask, [b * n_views, b * n_views]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        mask = mask.repeat(anchor_count, contrast_count)  # positive index

        # mask-out self-contrast cases in all diagonal positions; dig to 0, others to 1 (negative samples)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask  # positive samples except diagonal elements

        # compute log_prob, except the diagonal positions, [b * n_views, b * n_views]
        exp_logits = torch.exp(similarity_matrix) * logits_mask  # exp(z_i * z_a / T)

        # SupCon out
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # sup_out

        # loss, [n_views, bsz]
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class TNCLoss(nn.Module):
    def __init__(self, args, idx=None):
        super(TNCLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["TNC"]
        self.modalities = args.dataset_config["modality_names"]
        self.w = self.config["weight"]

        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, disc_features, neighbors):
        disc_pos, disc_neg = disc_features
        pos_neighbors, neg_neighbors = neighbors
        p_loss = self.bce_loss(disc_pos, pos_neighbors)
        n_loss = self.bce_loss(disc_neg, neg_neighbors)
        n_loss_u = self.bce_loss(disc_neg, neg_neighbors)
        loss = (p_loss + self.w * n_loss_u + (1 - self.w) * n_loss) / 2

        return loss


class CocoaLoss(nn.Module):
    def __init__(self, args, idx=None):
        """Cocoa loss similar to CMC
        Reference: https://github.com/cruiseresearchgroup/COCOA/blob/main/src/losses.py
        Paper: https://dl.acm.org/doi/10.1145/3550316
        """
        super(CocoaLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["Cocoa"]
        self.modalities = args.dataset_config["modality_names"]

        self.temperature = self.config["temperature"]
        self.scale_loss = self.config["scale_loss"]
        self.lambd = self.config["lambd"]

    def calc_loss(self, mod_features):
        """
        mod_features: [mod, batch, channel]
        """
        mods, batch, channel = mod_features.shape

        # Positive Pairs
        sample_features = mod_features.permute(1, 0, 2)
        sim = torch.matmul(sample_features, sample_features.permute(0, 2, 1))
        sim = torch.ones((batch, mods, mods)).to(self.args.device).float() - sim
        sim = torch.exp(sim / self.temperature)
        pos_error = torch.mean(sim)

        # Negative Pairs
        sim = torch.matmul(mod_features, mod_features.permute(0, 2, 1))
        sim = torch.exp(sim / self.temperature)
        tri_mask = torch.ones([batch, batch]).to(self.args.device).bool()
        tri_mask.fill_diagonal_(False).unsqueeze(0).repeat(mods, 1, 1)
        off_diag_sim = torch.masked_select(sim, tri_mask).reshape([mods, batch, batch - 1])
        neg_error = torch.sum(torch.mean(off_diag_sim, dim=[1, 2]), dim=0)

        loss = self.scale_loss * torch.mean(pos_error) + self.lambd * torch.mean(neg_error)

        return loss

    def forward(self, mod_features):
        features = torch.stack([mod_features[mod] for mod in self.modalities], dim=0)
        features_norm = F.normalize(features, dim=-1)
        loss = self.calc_loss(features_norm)

        return loss


class TS2VecLoss(nn.Module):
    def __init__(self, args):
        """TS2Vec loss
        https://github.com/yuezhihan/ts2vec
        """
        super(TS2VecLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["TS2Vec"]
        self.modalities = args.dataset_config["modality_names"]
        self.temperature = args.dataset_config[args.learn_framework]["temperature"]
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=-1)

    def mask_correlated_samples(self, seq_len=None, batch_size=0):
        """
        Return a mask where the correlated sample locations are 0.
        Output: [seq, 2b, 2b]
        """
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        if seq_len is not None:
            mask = mask.unsqueeze(0).repeat(seq_len, 1, 1)
        return mask

    def hierarchical_contrastive_loss(self, z1, z2, alpha=0.5, temporal_unit=0):
        loss = torch.tensor(0.0, device=z1.device)
        d = 0
        while z1.size(1) > 1:
            if alpha != 0:
                loss += alpha * self.instance_contrastive_loss(z1, z2)
            if d >= temporal_unit:
                if 1 - alpha != 0:
                    loss += (1 - alpha) * self.temporal_contrastive_loss(z1, z2)
            d += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        if z1.size(1) == 1:
            if alpha != 0:
                loss += alpha * self.instance_contrastive_loss(z1, z2)
            d += 1
        return loss / d

    # def instance_contrastive_loss(self, mod1_feature, mod2_feature, idx=None):
    #     """
    #     TS2Vec instance wise contrastive loss, same as SimCLR
    #     """
    #     batch_size = mod1_feature.shape[0]
    #     N = 2 * batch_size
    #     # Calculate similarity
    #     z = torch.cat((mod1_feature, mod2_feature), dim=0)
    #     sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
    #     sim_i_j = torch.diag(sim, batch_size)
    #     sim_j_i = torch.diag(sim, -batch_size)

    #     positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    #     negative_samples = sim[self.mask_correlated_samples(None, batch_size)].reshape(N, -1)

    #     labels = torch.zeros(N).to(positive_samples.device).long()
    #     logits = torch.cat((positive_samples, negative_samples), dim=1)
    #     loss = self.criterion(logits, labels) / N
    #     return loss

    # def temporal_contrastive_loss(self, mod1_feature, mod2_feature):
    #     """
    #     TS2Vec temporal wise contrastive loss,
    #     cross transformation feature time sample considered as positive
    #     other time samples within a seq are considered as negative
    #     """

    #     # we want to compare each sequence
    #     # [b, seq, dim]
    #     mod1_seq_features = mod1_feature
    #     mod2_seq_features = mod2_feature
    #     batch_size, seq_len, dim = mod1_seq_features.shape

    #     N = 2 * seq_len

    #     # Calculate similarity
    #     z = torch.cat((mod1_seq_features, mod2_seq_features), dim=1)
    #     sim = self.similarity_f(z.unsqueeze(2), z.unsqueeze(1)) / self.temperature
    #     sim_i_j = torch.diagonal(sim, seq_len, dim1=-2, dim2=-1)
    #     sim_j_i = torch.diagonal(sim, -seq_len, dim1=-2, dim2=-1)

    #     positive_samples = torch.cat((sim_i_j, sim_j_i), dim=1).reshape(batch_size, N, 1)
    #     negative_samples = sim[self.mask_correlated_samples(batch_size, seq_len)].reshape(batch_size, N, -1)

    #     # Compute loss
    #     labels = torch.zeros(batch_size * N).to(positive_samples.device).long()
    #     logits = torch.cat((positive_samples, negative_samples), dim=2).reshape(batch_size * N, -1)
    #     contrastive_loss = self.criterion(logits, labels)

    #     return contrastive_loss
    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        if B == 1:
            return z1.new_tensor(0.0)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def temporal_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        if T == 1:
            return z1.new_tensor(0.0)
        z = torch.cat([z1, z2], dim=1)  # B x 2T x C
        sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        return loss

    def forward(self, mod_features1, mod_features2, idx=None):
        """
        loss = instance contrastive loss + temporal contrastive loss
        Procedure:
            (1) For each batch, compute the instance contrastive loss
            (2) Split input into time sequences
            (3) For each time squence, compute the temporal contrastive loss
        """
        batch_size = mod_features1.shape[0]
        seq_len = self.args.dataset_config["seq_len"]

        # step 0: split features into subsequence B, T, C
        split_mod_features1 = mod_features1.reshape(batch_size // seq_len, seq_len, -1)
        split_mod_features2 = mod_features2.reshape(batch_size // seq_len, seq_len, -1)

        loss = self.hierarchical_contrastive_loss(split_mod_features1, split_mod_features2)
        # step 1: Instance wise contrastive loss
        # instance_contrastive_loss = self.instance_contrastive_loss(split_mod_features1, split_mod_features2)

        # step 2.2 calculate temporal contrastive loss
        # temporal_contrastive_loss = self.temporal_contrastive_loss(split_mod_features1, split_mod_features2)

        # loss = instance_contrastive_loss + temporal_contrastive_loss
        return loss


class GMCLoss(nn.Module):
    """GMC Loss function
    https://arxiv.org/pdf/2202.03390.pdf
    """

    def __init__(self, args):
        super(GMCLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["GMC"]
        self.batch_size = args.batch_size
        self.modalities = args.dataset_config["modality_names"]
        self.temperature = args.dataset_config[args.learn_framework]["temperature"]

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def infonce(self, mod_features):
        batch_representations = [mod_features[mod] for mod in self.modalities]
        batch_representations.append(mod_features["joint"])

        batch_size = batch_representations[0].shape[0]
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat([batch_representations[-1], batch_representations[mod]], dim=0)
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / self.temperature)
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod) - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(batch_representations[-1] * batch_representations[mod], dim=-1) / self.temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1))
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        return loss

    def infonce_with_joints_as_negatives(self, mod_features):
        """ """
        batch_representations = [mod_features[mod] for mod in self.modalities]
        batch_representations.append(mod_features["joint"])

        batch_size = batch_representations[0].shape[0]
        # Similarity among joints, [B, B]
        sim_matrix_joints = torch.exp(
            torch.mm(batch_representations[-1], batch_representations[-1].t().contiguous()) / self.temperature
        )
        # Mask out the diagonals, [B, B]
        mask_joints = (
            torch.ones_like(sim_matrix_joints) - torch.eye(batch_size, device=sim_matrix_joints.device)
        ).bool()
        # Remove diagonals and resize, [B, B-1]
        sim_matrix_joints = sim_matrix_joints.masked_select(mask_joints).view(batch_size, -1)

        # compute loss - for each pair joints-modality
        # Cosine loss on positive pairs
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            pos_sim_joint_mod = torch.exp(
                torch.sum(batch_representations[-1] * batch_representations[mod], dim=-1) / self.temperature
            )
            loss_joint_mod = -torch.log(pos_sim_joint_mod / sim_matrix_joints.sum(dim=-1))
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        return loss

    def forward(self, mod_features):
        return self.infonce(mod_features)


class MAELoss(nn.Module):
    """MAE Loss function
    https://github.com/facebookresearch/mae/blob/main/models_mae.py
    """

    def __init__(self, args):
        super(MAELoss, self).__init__()
        self.args = args
        self.modalities = args.dataset_config["modality_names"]
        self.backbone_config = args.dataset_config[args.model]
        self.generative_config = args.dataset_config["MAE"]
        self.locations = args.dataset_config["location_names"]
        self.norm_pix_loss = True

        # initialize loss weights
        self.initalize_loss_weights(weighting_scheme="linear")

        # psnr
        self.psnr = PeakSignalNoiseRatio()

    def patchify(self, imgs, patch_size, in_channel):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        ph, pw = patch_size

        h = imgs.shape[2] // ph
        w = imgs.shape[3] // pw

        x = imgs.reshape(shape=(imgs.shape[0], in_channel, h, ph, w, pw))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(imgs.shape[0], h * w, ph * pw * in_channel)
        return x
    
    def unpatchify(self, x, patch_size, in_channel, imgs_shape):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        ph, pw = patch_size

        h = imgs_shape[2] // ph
        w = imgs_shape[3] // pw

        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, in_channel))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_channel, h * ph, w * pw))
        
        return imgs

    def forward(self, padded_x, decoded_x, masks):
        total_loss = 0
        mod_weights = {}
        if self.args.dataset == "ACIDS":
            mod_weights['seismic'] = 0.5
            mod_weights['audio'] = 1.5
        else:
            for mod in self.modalities:
                mod_weights[mod] = 1.0
                
        for loc in self.locations:
            # calculate the energy of each signal
            target_energy = self.total_energy_modalities(padded_x[loc])
            for mod in self.modalities:
                mask = masks[loc][mod] # (128,256)
                pred = decoded_x[loc][mod] # (128,256,32)
                
                if self.generative_config["loss_scheme"] == "linear-weighted":
                    # Linear weighted MAE loss
                    if self.args.model == "TransformerV4":
                        # unpatchify the pred before computing loss weight
                        in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                        # imgs shape
                        target = padded_x[loc][mod]
                        imgs_shape = target.shape
                        pred = self.unpatchify(pred, self.backbone_config["patch_size"]["freq"][mod], in_channel, imgs_shape)
                    else:
                        target = padded_x[loc][mod]
                        pred = pred
                    # compute loss weight
                    loss_weight = self.lossWeighingHelper(pred, target, scheme="linear")
                    
                    if self.norm_pix_loss:
                        mean = target.mean(dim=-1, keepdim=True)
                        var = target.var(dim=-1, keepdim=True)
                        target = (target - mean) / (var + 1.0e-6) ** 0.5

                    loss = (pred - target) ** 2 # [B, C, H, W]
                    loss = loss * loss_weight # [B, C, H, W]
                    
                    # patchify the loss
                    in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                    if self.args.model == "TransformerV4":
                        patch_size = self.backbone_config["patch_size"]["freq"][mod]
                    else:
                        patch_size = self.generative_config["patch_size"][mod]
                    loss = self.patchify(loss, patch_size, in_channel)    
                    
                    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
                    
                    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                    total_loss += loss
                    
                if self.generative_config["loss_scheme"] == "linear-mod-weighted":
                    # Linear weighted MAE loss
                    if self.args.model == "TransformerV4":
                        # unpatchify the pred before computing loss weight
                        in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                        # imgs shape
                        target = padded_x[loc][mod]
                        imgs_shape = target.shape
                        pred = self.unpatchify(pred, self.backbone_config["patch_size"]["freq"][mod], in_channel, imgs_shape)
                    else:
                        target = padded_x[loc][mod]
                        pred = pred
                    # compute loss weight
                    loss_weight = self.lossWeighingHelper(pred, target, scheme="linear")
                    
                    if self.norm_pix_loss:
                        mean = target.mean(dim=-1, keepdim=True)
                        var = target.var(dim=-1, keepdim=True)
                        target = (target - mean) / (var + 1.0e-6) ** 0.5

                    loss = (pred - target) ** 2 # [B, C, H, W]
                    loss = loss * loss_weight # [B, C, H, W]
                    
                    # patchify the loss
                    in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                    if self.args.model == "TransformerV4":
                        patch_size = self.backbone_config["patch_size"]["freq"][mod]
                    else:
                        patch_size = self.generative_config["patch_size"][mod]
                    loss = self.patchify(loss, patch_size, in_channel)    
                    
                    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
                    
                    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                    loss = loss * mod_weights[mod]
                    total_loss += loss
                    
                if self.generative_config["loss_scheme"] == "wpsnr-v4":
                    # Linear weighted MAE loss
                    if self.args.model == "TransformerV4":
                        # unpatchify the pred before computing loss weight
                        in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                        # imgs shape
                        target = padded_x[loc][mod]
                        imgs_shape = target.shape
                        pred = self.unpatchify(pred, self.backbone_config["patch_size"]["freq"][mod], in_channel, imgs_shape)
                    else:
                        target = padded_x[loc][mod]
                        pred = pred
                    # compute loss weight
                    loss_weight = self.lossWeighingHelper(pred, target, scheme="linear")
                    
                    if self.norm_pix_loss:
                        unnormalized_target = target.clone()
                        mean = target.mean(dim=-1, keepdim=True)
                        var = target.var(dim=-1, keepdim=True)
                        target = (target - mean) / (var + 1.0e-6) ** 0.5

                    loss = (pred - target) ** 2 # [B, C, H, W]
                    loss = loss * loss_weight # [B, C, H, W]
                    
                    # patchify the loss
                    in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                    if self.args.model == "TransformerV4":
                        patch_size = self.backbone_config["patch_size"]["freq"][mod]
                    else:
                        patch_size = self.generative_config["patch_size"][mod]
                    
                    # calculate masked weighted mse loss
                    loss = self.patchify(loss, patch_size, in_channel)
                    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
                    energy = self.total_energy(unnormalized_target)
                    loss = (energy / (loss * mask).sum(-1)) / mask.sum()
                    
                    # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                    # mse = loss
                    
                    # use target energy
                    psnr = 10 * torch.log10(loss)
                    psnr = psnr.sum()
                    # psnr = 10 * torch.log10(self.total_energy(target) / mse)

                    
                    # use masked portion of target energy instead
                    # target = self.patchify(target, patch_size, in_channel)
                    # masked_target = self.apply_mask_to_patches(target, mask)
                    # psnr = 10 * torch.log10(self.total_energy(masked_target) / mse)
                    
                    # use energy portion instead
                    # energy_ratio = self.total_energy(masked_target) / self.total_energy(target)
                    # psnr = 10 * torch.log10(energy_ratio / mse)
                    
                    total_loss -= psnr
                    
                elif self.generative_config["loss_scheme"] == "psnr":
                    # PSNR loss
                    if self.args.model == "TransformerV4":
                        in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                        target = self.patchify(
                            padded_x[loc][mod],
                            self.backbone_config["patch_size"]["freq"][mod],
                            in_channel,
                        )
                    else:
                        patch_size = self.generative_config["patch_size"][mod]
                        patch_area = patch_size[0] * patch_size[1]
                        in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                        target = self.patchify(
                            padded_x[loc][mod],
                            patch_size,
                            in_channel,
                        )

                        pred = self.patchify(
                            pred,
                            patch_size,
                            in_channel,
                        )
                        mask = mask.reshape((pred.shape[0], -1))

                    pred_masked = self.apply_mask_to_patches(pred, mask)
                    target_masked = self.apply_mask_to_patches(target, mask)
                    
                    # loss = (pred - target) ** 2 # [B, C, H, W]
                    # loss = loss * loss_weight # [B, C, H, W]
                    loss = -peak_signal_noise_ratio(pred_masked, target_masked, reduction=None)
                    loss = loss/ mask.sum()  # mean loss on removed patches
                    total_loss += loss
                
                elif self.generative_config["loss_scheme"] == "wpsnr":
                    # WPSNR Loss
                    if self.args.model == "TransformerV4":
                        # unpatchify the pred before computing loss weight
                        in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                        # imgs shape
                        target = padded_x[loc][mod]
                        imgs_shape = target.shape
                        pred = self.unpatchify(pred, self.backbone_config["patch_size"]["freq"][mod], in_channel, imgs_shape)
                    else:
                        target = padded_x[loc][mod]
                        pred = pred
                    
                    # compute loss weight
                    loss_weight = self.lossWeighingHelper(pred, target, scheme="linear")
                    
                    # compute target energy using magnitude spectra
                    # target_energy = self.total_energy(target)
                    
                    # compute weighted mse loss
                    loss = (pred - target) ** 2 # [B, C, H, W]
                    loss = loss * loss_weight # [B, C, H, W]
                    
                    # do masking on the loss, patchify the loss for masking
                    in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                    if self.args.model == "TransformerV4":
                        patch_size = self.backbone_config["patch_size"]["freq"][mod]
                    else:
                        patch_size = self.generative_config["patch_size"][mod]
                    loss = self.patchify(loss, patch_size, in_channel) 
                    loss = self.apply_mask_to_patches(loss, mask)
                    
                    # unpatchify the loss
                    imgs_shape = target.shape
                    loss = self.unpatchify(loss, self.backbone_config["patch_size"]["freq"][mod], in_channel, imgs_shape)
                    
                    # accumulate loss across all but batch dimension
                    loss = loss.sum(dim=(1,2,3))
                    
                    # start calculating wpsnr
                    n_obs = torch.tensor(target.numel(), device=target.device)
                    psnr_base_e = 0.3  * torch.log(target_energy) - torch.log(loss) # log energy scaling can be done here
                    psnr_vals = psnr_base_e * (10 / torch.log(torch.tensor(10.)))
                    
                    # loss = -self.weighted_peak_to_noise_ratio(pred_masked, target_masked, loss_weight)
                    loss = -psnr_vals.sum() / mask.sum()  # mean loss on removed patches
                    total_loss += loss
                
                elif self.generative_config["loss_scheme"] == "wpsnr-v2":
                    # WPSNR Loss
                    if self.args.model == "TransformerV4":
                        # unpatchify the pred before computing loss weight
                        in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                        # imgs shape
                        target = padded_x[loc][mod]
                        imgs_shape = target.shape
                        pred = self.unpatchify(pred, self.backbone_config["patch_size"]["freq"][mod], in_channel, imgs_shape)
                    else:
                        target = padded_x[loc][mod]
                        pred = pred
                    
                    # compute loss weight
                    loss_weight = self.lossWeighingHelper(pred, target, scheme="linear")
                    
                    # compute target energy using magnitude spectra
                    # target_energy = self.total_energy(target)
                    
                    # compute weighted mse loss
                    loss = (pred - target) ** 2 # [B, C, H, W]
                    loss = loss * loss_weight # [B, C, H, W]
                    
                    # do masking on the loss, patchify the loss for masking
                    in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                    if self.args.model == "TransformerV4":
                        patch_size = self.backbone_config["patch_size"]["freq"][mod]
                    else:
                        patch_size = self.generative_config["patch_size"][mod]
                    loss = self.patchify(loss, patch_size, in_channel) 
                    loss = self.apply_mask_to_patches(loss, mask)
                    
                    # unpatchify the loss
                    imgs_shape = target.shape
                    loss = self.unpatchify(loss, self.backbone_config["patch_size"]["freq"][mod], in_channel, imgs_shape)
                    
                    # accumulate loss across all but batch dimension
                    loss = loss.sum(dim=(1,2,3))
                    
                    # start calculating wpsnr
                    # n_obs = torch.tensor(target.numel(), device=target.device)
                    # psnr_base_e = torch.log(target_energy) * loss
                    psnr_base_e = target_energy * loss
                    psnr_vals = psnr_base_e * (10 / torch.log(torch.tensor(10.)))
                    
                    # loss = -self.weighted_peak_to_noise_ratio(pred_masked, target_masked, loss_weight)
                    loss = psnr_vals.sum() / mask.sum()  # mean loss on removed patches
                    total_loss += loss
                
                elif self.generative_config["loss_scheme"] == "weighted-psnr-old":
                    # Linear weighted MAE loss
                    if self.args.model == "TransformerV4":
                        # unpatchify the pred before computing loss weight
                        in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                        # imgs shape
                        target = padded_x[loc][mod]
                        imgs_shape = target.shape
                        pred = self.unpatchify(pred, self.backbone_config["patch_size"]["freq"][mod], in_channel, imgs_shape)
                    else:
                        target = padded_x[loc][mod]
                        pred = pred

                    ## start psnr loss
                    sse = (target - pred) ** 2
                    # compute loss weight
                    loss_weight = self.lossWeighingHelper(pred, target, scheme="linear")
                    loss = sse * loss_weight
                    
                    # patchify the loss for masking
                    in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                    if self.args.model == "TransformerV4":
                        patch_size = self.backbone_config["patch_size"]["freq"][mod]
                    else:
                        patch_size = self.generative_config["patch_size"][mod]
                    loss = self.patchify(loss, patch_size, in_channel)    
                    
                    # mask out the loss for visible patches, and calculate mse
                    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
                    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                    
                    # do the rest
                    max_val = torch.max(target)
                    loss = -10 * torch.log10(max_val ** 2 / loss) # flip the sign
                    
                    total_loss += loss
                
                elif self.generative_config["loss_scheme"] == "reconstruction":
                    # regular MAE loss
                    if self.args.model == "TransformerV4":
                        in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                        target = self.patchify(
                            padded_x[loc][mod],
                            self.backbone_config["patch_size"]["freq"][mod],
                            in_channel,
                        )
                    else:
                        patch_size = self.generative_config["patch_size"][mod]
                        patch_area = patch_size[0] * patch_size[1]
                        in_channel = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                        target = self.patchify(
                            padded_x[loc][mod],
                            patch_size,
                            in_channel,
                        )

                        pred = self.patchify(
                            pred,
                            patch_size,
                            in_channel,
                        )
                        mask = mask.reshape((pred.shape[0], -1))
                        
                    if self.norm_pix_loss:
                        mean = target.mean(dim=-1, keepdim=True)
                        var = target.var(dim=-1, keepdim=True)
                        target = (target - mean) / (var + 1.0e-6) ** 0.5

                    loss = (pred - target) ** 2 # [128, 256, 32]
                    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
                

                    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                    total_loss += loss

        return total_loss
    
    def lossWeighingHelper(self, pred, target, scheme="linear", patchify=False):
        """
        Return a loss weighting tensor, to be multiplied by the loss tensor
        pred: (N, C, H, W)
        target: (N, C, H, W)
        """
        # TODO: you do not need pred here or unpatchify before this function, if you are using the linear scheme or pure harmonic scheme
        # scheme = "pure_harmonic"
        if scheme == "linear":
            loss_dim = target.shape[-1]
            freq_bins = loss_dim
            half_freq_bins = freq_bins // 2
            weights = torch.cat((torch.linspace(1, 0, half_freq_bins), torch.linspace(0, 1, half_freq_bins))).unsqueeze(0).to(self.args.device)
            # set 0th entry to 0, DC component should not be weighted
            # weights[..., 0] = 0
        elif scheme == "pure_harmonic":
            loss_dim = target.shape[-1]
            freq_bins = loss_dim
            half_freq_bins = freq_bins // 2
            target_complex = self.to_complex(target)
        elif scheme == "coherence":
            # cross_coh = self.calculate_average_cross_coherence(target) # TODO: use cross coherence as weights, basically reconstruct information that is consistent across modality sensors
            cross_coh = self.calculate_average_cross_correlation(target)
            return cross_coh # directly return, size is handled in the function
            pass
            
        # Create a tensor of same size as target, with each element being the corresponding weight
        weights_tensor= torch.zeros_like(target).to(self.args.device)
        # set last dim values to weights
        weights_tensor[..., :weights.shape[-1]] = weights
        
        if patchify:
            pass
        return weights_tensor # [B, C, H, W]
    
    def calculate_cross_coherence(self, input_tensor):
        """Calculate the cross-coherence of each signal pair in the input tensor.
        input_tensor: [b, c, i, s]
        coherence_tensor: [b, c//2, i]
        """
        b, c, i, s = input_tensor.shape
        assert c % 2 == 0, "Number of channels must be even (magnitude and phase for each signal)"

        # Initialize an empty list to store the coherence tensors
        coherence_list = []

        # Iterate over each pair of channels
        for ch in range(0, c, 2):
            # Convert magnitude and phase to complex Fourier transform
            X = input_tensor[:, ch, :, :] * torch.exp(1j * input_tensor[:, ch+1, :, :])
            Y = input_tensor[:, ch+2, :, :] * torch.exp(1j * input_tensor[:, ch+3, :, :])

            # Calculate spectral densities
            Pxx = torch.mean(X * torch.conj(X), dim=-1)
            Pyy = torch.mean(Y * torch.conj(Y), dim=-1)
            Pxy = torch.mean(X * torch.conj(Y), dim=-1)

            # Calculate coherence
            coherence = torch.abs(Pxy) ** 2 / (Pxx * Pyy)
            coherence_list.append(coherence)

        # Stack the coherence tensors along the channel dimension
        coherence_tensor = torch.stack(coherence_list, dim=1)
        
        # Average over the time dimension frames
        coherence_tensor = torch.mean(coherence_tensor, dim=-1)

        return coherence_tensor

    def calculate_average_cross_coherence(self, input_tensor):
        """Calculate the average cross-coherence of each channel with all other channels in the input tensor.
        input_tensor: [b, c, i, s]
        coherence_tensor: [b, c//2, i, s]
        """
        b, c, i, s = input_tensor.shape
        assert c % 2 == 0, "Number of channels must be even (magnitude and phase for each signal)"

        # Initialize an empty list to store the coherence tensors
        coherence_list = []

        # Iterate over each pair of channels
        for ch in range(0, c, 2):
            coherence_sum = torch.zeros(b, i, s, device=input_tensor.device, dtype=torch.complex64)
            count = 0

            X = input_tensor[:, ch, :, :] * torch.exp(1j * input_tensor[:, ch+1, :, :])
            X = X[0,0,:]
            Pxx = torch.conj(X) * X

            # Calculate coherence with all other channels
            for ch_other in range(0, c, 2):
                if ch_other == ch:  # skip coherence with itself
                    continue

                Y = input_tensor[:, ch_other, :, :] * torch.exp(1j * input_tensor[:, ch_other+1, :, :])
                Y = Y[0,0,:]
                Pyy = torch.conj(Y) * Y
                Pxy = torch.conj(X) * Y

                # Calculate coherence
                
                # nominator = torch.abs(Pxy) ** 2
                # denominator = torch.abs(Pxx * Pyy)
                # coherence = torch.div(nominator , denominator, rounding_mode = None)
                
                coherence = torch.abs(Pxy)**2 / Pxx.real / Pyy.real
                coherence_sum += coherence
                count += 1

            # Calculate average coherence
            average_coherence = coherence_sum / count
            coherence_list.append(average_coherence)

        # Stack the average coherence tensors along the channel dimension
        average_coherence_tensor = torch.stack(coherence_list, dim=1)

        # Fix the NaN values
        average_coherence_tensor = torch.nan_to_num(torch.abs(average_coherence_tensor))

        # from b,c//2,i,s to b,c,i,s by repeating the values
        average_coherence_tensor = average_coherence_tensor.repeat(1,2,1,1)
        
        return average_coherence_tensor
    
    def calculate_average_cross_correlation(self, input_tensor):
        """Calculate the average cross-correlation of each channel with all other channels in the input tensor.
        input_tensor: [b, c, i, s]
        cross_correlation_tensor: [b, c//2, i, s]
        """
        b, c, i, s = input_tensor.shape
        assert c % 2 == 0, "Number of channels must be even (magnitude and phase for each signal)"

        # Initialize a list to store the cross-correlation tensors
        cross_correlation_list = []

        # Iterate over each pair of channels
        for ch in range(0, c, 2):
            cross_correlation_sum = torch.zeros((b, i, s), device=input_tensor.device)
            count = 0

            X = input_tensor[:, ch, :, :]

            # Calculate cross-correlation with all other channels
            for ch_other in range(0, c, 2):
                if ch_other == ch:  # skip cross-correlation with itself
                    continue

                Y = input_tensor[:, ch_other, :, :]

                # Calculate cross-correlation
                X_mean = torch.mean(X, dim=-1, keepdim=True).expand_as(X)
                Y_mean = torch.mean(Y, dim=-1, keepdim=True).expand_as(Y)
                X_std = torch.std(X, dim=-1, keepdim=True).expand_as(X)
                Y_std = torch.std(Y, dim=-1, keepdim=True).expand_as(Y)
                
                cross_correlation = ((X - X_mean) * (Y - Y_mean)) / (X_std * Y_std)

                cross_correlation_sum += cross_correlation
                count += 1

            # Calculate average cross-correlation
            average_cross_correlation = cross_correlation_sum / count
            cross_correlation_list.append(average_cross_correlation)

        # Stack the average cross-correlation tensors along the new channel dimension
        average_cross_correlation_tensor = torch.stack(cross_correlation_list, dim=1)

        # Normalize to 0-1
        average_cross_correlation_tensor = (average_cross_correlation_tensor + 1) / 2.0
        
        # Fix the NaN values
        average_cross_correlation_tensor = torch.abs(torch.nan_to_num(average_cross_correlation_tensor))
        
        # from b,c//2,i,s to b,c,i,s by repeating the values
        average_cross_correlation_tensor = average_cross_correlation_tensor.repeat(1,2,1,1)
        return average_cross_correlation_tensor


    def weightedPSNR(self, pred, target, mask = None): # older implementation
        """
        pred: (N, C, H, W)
        target: (N, C, H, W)
        weights: (N, C, H, W)
        """
        sse = (target - pred) ** 2
        # compute loss weight
        loss_weight = self.lossWeighingHelper(pred, target, scheme="linear")
        sse = sse * loss_weight
        # mask out the loss
        if mask is not None:
            sse = sse * mask
        # do the rest
        mse = torch.mean(sse)
        max_val = torch.max(target)
        weighted_psnr = 10 * torch.log10(max_val ** 2 / mse)
        return weighted_psnr
    
    def psnr(self, target, pred):
        mse = torch.mean((target - pred) ** 2)
        max_val = torch.max(target)
        psnr = 10 * torch.log10(max_val ** 2 / mse)
        return psnr
    
    def weighted_peak_to_noise_ratio(self, pred, target, loss_weight=1):
        # data_range = target.max() - target.min()
        target_energy = self.total_energy(target)
        squared_error = torch.pow(pred - target, 2)
        # weight the squared error
        squared_error = squared_error * loss_weight
        sum_squared_error = torch.sum(squared_error)
        n_obs = torch.tensor(target.numel(), device=target.device)
        # psnr_base_e = 2 * torch.log(target_energy) - torch.log(sum_squared_error / n_obs)
        psnr_base_e = torch.log(target_energy) - torch.log(sum_squared_error / n_obs)
        psnr_vals = psnr_base_e * (10 / torch.log(torch.tensor(10.0)))
        return reduce(psnr_vals, reduction=None)
    
    def total_energy(self,input_tensor):
        """Calculate the energy of each signal in the input tensor.
        input_tensor: [b, c, i, s]
        energy_tensor: [b, c//2]
        """
        # TODO: Use normalized energy tensors across runs/modalities? Here, seismic energy may dominate acoustic
        b, c, i, s = input_tensor.shape
        assert c % 2 == 0, "Number of channels must be even (magnitude and phase for each signal)"

        # Initialize an empty list to store the energy tensors
        energy_list = []

        # Iterate over each pair of channels
        for ch in range(0, c, 2):
            # Calculate the magnitude for the pair, square it to get the energy and sum across the frames
            energy = torch.sum(input_tensor[:, ch, :, :] ** 2, dim=(-1, -2))
            energy_list.append(energy)

        # Stack the energy tensors along the channel dimension
        energy_tensor = torch.stack(energy_list, dim=1) # b, c//2
        
        # Sum the energy across all physical channels
        energy_tensor = torch.sum(energy_tensor, dim=1) # b
        
        return energy_tensor
    
    def total_energy_modalities(self, location_tensors):
        energy_tensors = []
        for loc in self.modalities:
            input_tensor = location_tensors[loc]
            energy_tensor = self.total_energy(input_tensor)
            energy_tensors.append(energy_tensor)
        # stack the energy tensors along the channel dimension
        energy_tensor = torch.stack(energy_tensors, dim=1) # b, c//2
        # take the mean of the energy tensors
        energy_tensor = torch.mean(energy_tensor, dim=1) # b
        return energy_tensor
    
    def apply_mask_to_patches(self, patches, mask):
        # patches shape: (N, L, patch_size**2 *3)
        # mask shape: (N, L)

        # Extend mask dimensions
        mask = mask.unsqueeze(-1)  # mask shape becomes (N, L, 1)
        
        # Expand mask to match patches dimensions
        mask = mask.expand_as(patches)  # mask shape becomes (N, L, patch_size**2 *3)

        # Apply mask
        masked_patches = patches * mask

        return masked_patches

    def to_complex(self,x):
        # x shape: (N, C, H, W), C = 2 * num_complex_channels
        
        N, C, H, W = x.shape
        num_complex_channels = C // 2

        # reshape to put real and imaginary parts together
        x_complex = x.view(N, num_complex_channels, 2, H, W)

        # switch the complex dimension to the last
        x_complex = x_complex.permute(0, 1, 3, 4, 2)

        # convert to complex
        x_complex = x_complex.contiguous() #.view(N, num_complex_channels, H, W)
        x_complex = torch.view_as_complex(x_complex)

        return x_complex
    def to_time_domain(self, x_complex):
        # x_complex shape: (N, num_complex_channels, H, W)

        # perform ifft
        x_time = torch.fft.ifft(x_complex)

        return x_time

    def initalize_loss_weights(self, weighting_scheme= "linear"):

        if weighting_scheme == "linear":
            # linear weighted scheme
            # Parkland 288
            freq_bins = 576 # loss.shape[-1]
            half_freq_bins = freq_bins // 2
            self.weights_576 = torch.cat((torch.linspace(0, 1, half_freq_bins), torch.linspace(1, 0, half_freq_bins))).unsqueeze(0).to(self.args.device)
            self.weights_288 = torch.cat((torch.linspace(0, 1, half_freq_bins // 2), torch.linspace(1, 0, half_freq_bins // 2))).unsqueeze(0).to(self.args.device)
            # ACIDS 256
            freq_bins = 256 # loss.shape[-1]
            half_freq_bins = freq_bins // 2
            self.weights_256 = torch.cat((torch.linspace(0, 1, half_freq_bins), torch.linspace(1, 0, half_freq_bins))).unsqueeze(0).to(self.args.device)

        elif weighting_scheme == "exponential":
            # exponential weighted scheme
            # Parkland 288
            freq_bins = 576 # loss.shape[-1]
            half_freq_bins = freq_bins // 2
            exp_weights = torch.exp(torch.linspace(2, 0, half_freq_bins))
            self.weights_576 = torch.cat((exp_weights.flip(dims=[0]), exp_weights)).unsqueeze(0).to(self.args.device)
            self.weights_288 = torch.cat((exp_weights.flip(dims=[0])[:half_freq_bins // 2], exp_weights[:half_freq_bins // 2])).unsqueeze(0).to(self.args.device)
            # ACIDS 256
            freq_bins = 256 # loss.shape[-1]
            half_freq_bins = freq_bins // 2
            exp_weights = torch.exp(torch.linspace(2, 0, half_freq_bins))
            self.weights_256 = torch.cat((exp_weights.flip(dims=[0]), exp_weights)).unsqueeze(0).to(self.args.device)



class TSTCCLoss(nn.Module):
    """TS-TC Loss function
    https://github.com/emadeldeen24/TS-TCC/blob/main/trainer/trainer.py
    """

    def __init__(self, args):
        super(TSTCCLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["TSTCC"]

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        self.lambda1 = self.config["lambda1"]
        self.lambda2 = self.config["lambda2"]

        self.temperature = self.config["temperature"]

    def mask_correlated_samples(self, batch_size):
        """
        Mark diagonal elements and elements of upper-right and lower-left diagonals as 0.
        """
        N = 2 * batch_size
        diag_mat = torch.eye(batch_size).to(self.args.device)
        mask = torch.ones((N, N)).to(self.args.device)

        mask = mask.fill_diagonal_(0)
        mask[0:batch_size, batch_size : 2 * batch_size] -= diag_mat
        mask[batch_size : 2 * batch_size, 0:batch_size] -= diag_mat

        mask = mask.bool()

        return mask

    def forward_contrast(self, z_i, z_j, idx=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        # Calculate similarity
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / N

        return loss

    def forward(self, temporal_contrast_features, temporal_contrast_loss):
        tc_features1, tc_features2 = temporal_contrast_features
        loss = self.forward_contrast(tc_features1, tc_features2) * self.lambda2
        loss += (temporal_contrast_loss[0] + temporal_contrast_loss[1]) * self.lambda1
        return loss
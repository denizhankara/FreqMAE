import torch
import torch.nn as nn
import torch.nn.functional as F


class TNC(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone_config = backbone.config
        self.config = args.dataset_config["TNC"]

        # components
        self.backbone = backbone

        # Discriminator
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(2 * self.backbone_config["fc_dim"], 4 * self.backbone_config["fc_dim"]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4 * self.backbone_config["fc_dim"], 1),
        )

        torch.nn.init.xavier_uniform_(self.discriminator[0].weight)
        torch.nn.init.xavier_uniform_(self.discriminator[3].weight)

    def forward_disc(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.discriminator(x_all)
        return p.view((-1,))

    def forward(self, embedding):
        # get representation
        mod_features = self.backbone(embedding, class_head=False)

        b, dim = mod_features.shape
        seq_len = self.args.dataset_config["seq_len"]
        bs = b // seq_len

        # split into time features, [b, dim] -> [bs, sequence, dims]
        # mod feature shas shape b, dim
        # mod_samples = mod_features.repeat(seq_len, dim=0)
        split_mod_samples = mod_features.reshape(bs, seq_len, -1)

        # positive samples [bs, seq, dim] -> [bs, seq, seq, dim]
        split_pos_samples = split_mod_samples.unsqueeze(1).repeat_interleave(seq_len, dim=1)
        # positive masks [seq, seq] diagonal
        pos_mask = torch.eye(seq_len, dtype=bool)
        # get pos samples for label
        pos_samples = mod_features.repeat_interleave(seq_len - 1, dim=0)
        # diagonal are not considered, take the non diagonal entries, [bs, seq, seq - 1, dim] -> [b, seq - 1, dim]
        pos_features = split_pos_samples[:, ~pos_mask].view(b * (seq_len - 1), dim)

        # ompute negative samples
        # negative samples [bs, seq, dim] -> [bs, bs, seq, dim]
        split_neg_samples = split_mod_samples.unsqueeze(0).repeat_interleave(bs, dim=0)
        # negative masks [bs, bs]
        neg_mask = torch.eye(bs, dtype=bool)
        neg_samples = mod_features.repeat_interleave(bs - 1, dim=0)
        # diagonals are not considered, takes the non diagonal entries
        neg_features = split_neg_samples[~neg_mask].view(b * (bs - 1), dim)

        # get neighbor labels
        neighbors = torch.ones((len(pos_features))).to(self.args.device)
        # get non neighjbor labels
        non_neighbors = torch.zeros((len(neg_features))).to(self.args.device)

        # # nonlienar MLP

        # [b, dim] and [b, seq - 1, dim]
        disc_pos = self.forward_disc(pos_samples, pos_features)
        disc_neg = self.forward_disc(neg_samples, neg_features)

        return [disc_pos, disc_neg], [neighbors, non_neighbors]

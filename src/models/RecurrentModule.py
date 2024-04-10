import torch
import torch.nn as nn


class RecurrentBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers=2, dropout_ratio=0) -> None:
        """The initialization of the recurrent block."""
        super().__init__()

        self.gru = nn.GRU(
            in_channel, out_channel, num_layers, bias=True, batch_first=True, dropout=dropout_ratio, bidirectional=True
        )

    def forward(self, x):
        """The forward function of the recurrent block.
        TODO: Add mask such that only valid intervals are considered in taking the mean.
        Args:
            x (_type_): [b, c_in, intervals]
        Output:
            [b, c_out]
        """
        # [b, c, i] --> [b, i, c]
        x = x.permute(0, 2, 1)

        # GRU --> mean
        # [b, i, c] --> [b, i, c]
        output, hidden_output = self.gru(x)

        # [b, i, c] --> [b, c]
        output = torch.mean(output, dim=1)

        return output, hidden_output


class DecRecurrentBlock(nn.Module):
    def __init__(self, mod_interval, in_channel, out_channel, num_layers=2, dropout_ratio=0) -> None:
        """The initialization of the decoder recurrent block."""
        super().__init__()
        self.mod_interval = mod_interval
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_layers = num_layers

        self.mean_decoder = nn.Linear(self.out_channel, self.out_channel * mod_interval)

        self.gru = nn.GRU(
            out_channel,
            out_channel,
            num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_ratio,
            bidirectional=False,
        )

        self.dec_rnn_layer = nn.Linear(out_channel, in_channel)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden_state=None):
        """The forward function of the recurrent block.
        https://discuss.pytorch.org/t/about-bidirectional-gru-with-seq2seq-example-and-some-modifications/15588/4
        TODO: Add mask such that only valid intervals are considered in taking the mean.
        Args:
            x (_type_): [b, c_in, intervals]
        Output:
            [b, c_out]
        """
        # [b, bidirectional 2 * c] -> [b, c]
        mean_encoded_feature = x[:, : self.out_channel] + x[:, self.out_channel :]
        # [b, c] -> [b, i, c]
        # corresponds to the torch.mean decoder
        encoded_feature = self.mean_decoder(mean_encoded_feature)
        encoded_feature = encoded_feature.reshape(encoded_feature.shape[0], -1, self.out_channel)

        # since encoder has 2 * bidirectional = 4 layers of encoding, we only need last 2
        hidden_state = hidden_state[-self.num_layers :]
        dec_rnn_features, state = self.gru(encoded_feature, hidden_state)
        # GRU decode
        # soft max + linear dim
        dec_features = self.dec_rnn_layer(dec_rnn_features)
        dec_features = dec_features.permute(0, 2, 1)

        return dec_features

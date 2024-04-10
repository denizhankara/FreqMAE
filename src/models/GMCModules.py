import torch
import torch.nn as nn
import torch.nn.functional as F


class GMC(nn.Module):
    """
    https://arxiv.org/pdf/2202.03390.pdf
    """

    def __init__(self, args, backbone):
        super(GMC, self).__init__()

        self.args = args
        self.config = args.dataset_config["GMC"]
        self.model_config = args.dataset_config[args.model]
        self.locations = args.dataset_config["location_names"]
        self.modalities = args.dataset_config["modality_names"]

        # build encoders f(·) = {f1:M(·)}∪{f1(·), . . . , fM(·)}
        self.backbone = backbone

    def forward(self, freq_input):
        """
        Input:
            freq_input[loc][mod]
        Output:
            features {f1M(joint input), fm(single mod input) for m in 1 M}
        """

        return self.backbone(freq_input, class_head=False)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

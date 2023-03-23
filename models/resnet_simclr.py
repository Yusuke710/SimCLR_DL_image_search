import torch.nn as nn

import torch
from torchvision.models import resnet18, resnet50, ResNet50_Weights

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": resnet18(weights=None, num_classes=out_dim),
                            "resnet50": resnet50(weights=None, num_classes=out_dim),
                            # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
                            "resnet50_pretrain": resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)}

        self.backbone = self._get_basemodel(base_model)

        dim_mlp = self.backbone.fc.in_features
        # pre-trained model uses num class 1000. Modify in our case to compress vector
        self.backbone.fc = nn.Linear(dim_mlp, out_dim)

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)

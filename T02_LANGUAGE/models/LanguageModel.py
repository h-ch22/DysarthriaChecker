import torch.nn as nn

from models.ClassificationModel import ClassificationModel
from models.FeatureModel import FeatureModel


class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.feature_model = FeatureModel()
        self.classification_model = ClassificationModel()

    def forward(self, x):
        features = self.feature_model(x)
        outputs = self.classification_model(features)

        return outputs

import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, encoded_dim, num_classes):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(encoded_dim, num_classes),
        )
        
    def forward(self, representations):
        classification = self.classifier(representations)
        
        return classification

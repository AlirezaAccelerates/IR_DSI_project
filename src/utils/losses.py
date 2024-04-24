import torch
import torch.nn as nn


# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(ContrastiveLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output1, output2, label):
        # Calculate Euclidean distance between output1 and output2
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        # Calculate contrastive loss
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.epsilon - euclidean_distance, min=0.0), 2))

        # Return loss value
        return loss_contrastive
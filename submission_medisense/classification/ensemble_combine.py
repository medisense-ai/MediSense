import torch
import torch.nn as nn

from ensemble_LR import WeightedEnsembleModel


class CombinedEnsemble(nn.Module):
    def __init__(self, ensembles: nn.ModuleDict):
        super().__init__()

        self.ensembles = ensembles

    def forward(self, x):
        """
        x: dict with 'L' and 'R' keys, each containing a tensor of shape (batch_size, 1, 224, 224)
        """

        if not x.get("L") or not x.get("R"):
            raise ValueError("The input should have 'L' and 'R' keys.")

        if not len(x["L"]) == len(x["R"]) == 2:
            raise ValueError("The number of samples in 'L' and 'R' should equal to 2.")

        L_prob, R_prob = self.ensemble["L"](x["L"]), self.ensemble["R"](x["R"])

        # pick the class with the highest probability
        max_probs, _ = torch.stack([L_prob, R_prob], dim=0).max(dim=0)

        probs, labels = max_probs.max(dim=0)

        return probs, labels

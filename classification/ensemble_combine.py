import torch.nn as nn

from ensemble_LR import WeightedEnsembleModel

class CombinedEnsemble(nn.Module):
    def __init__(self, ensembles, detection_weights=None):
        super().__init__()

        self.ensemble_L = WeightedEnsembleModel(ensembles("L"), detection_weights.get("L"))
        self.ensemble_R = WeightedEnsembleModel(ensembles("R"), detection_weights.get("R"))
    
    def forward(self, x):
        '''
        x: dict with 'L' and 'R' keys, each containing a tensor of shape (batch_size, 1, 224, 224)
        '''

        if not x.get('L') or not x.get('R'):
            raise ValueError("The input should have 'L' and 'R' keys.")
        
        if not len(x['L']) == len(x['R']) == 2:
            raise ValueError("The number of samples in 'L' and 'R' should equal to 2.")

        L_prob, R_prob = self.ensemble_L(x['L']), self.ensemble_L(x['R'])

        # pick the class with the highest probability
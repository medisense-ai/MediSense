import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.metrics import precision_recall_fscore_support, f1_score, matthews_corrcoef


def _calculate_weights(metrics_list):
    """
    Accepts a list of a metric for each model in an ensemble and returns weights for their predictions.
    
    Args:
    metrics_list (list of float): A list of metric values for each model.

    Returns:
    list of float: The weights for each model's predictions.
    """
    
    # Ensure the metrics list is not empty
    if not metrics_list:
        raise ValueError("The metrics list should not be empty.")
    
    # Sum of all metrics to normalize the weights
    total_metric = sum(metrics_list)
    
    # Check if the total_metric is zero to avoid division by zero
    if total_metric == 0:
        raise ValueError("The sum of the metrics should not be zero.")
    
    # Calculate weights by normalizing each metric
    weights = [metric / total_metric for metric in metrics_list]
    
    return weights

def _calculate_weights_softmax(metrics_list):
    """
    Accepts a list of a metric for each model in an ensemble and returns weights for their predictions using softmax.
    
    Args:
    metrics_list (list of float): A list of metric values for each model.

    Returns:
    list of float: The weights for each model's predictions.
    """
    
    # Ensure the metrics list is not empty
    if not metrics_list:
        raise ValueError("The metrics list should not be empty.")
    
    # Convert the metrics list to a numpy array
    metrics_array = np.array(metrics_list)
    
    # Apply softmax to calculate weights
    exp_metrics = np.exp(metrics_array - np.max(metrics_array))  # Subtract max to prevent overflow
    weights = exp_metrics / exp_metrics.sum()
    
    return weights.tolist()

class WeightedEnsembleModel(nn.Module):
    def __init__(self, models, detection_weights=None):
        super().__init__()
        self.models_list= nn.ModuleList(models)

        num_models = len(self.models_list)
        if detection_weights is None:
            detection_weights = [1.0] * num_models

        self.detection_weights = detection_weights
        self.model_metrics = []

    def forward(self, x):
        if len(x) != len(self.models_list):
            raise ValueError("The number of inputs should match the number of models.")

        detect_outputs = []
        for pic, model, w_det in zip(x, self.models_list, self.detection_weights):
            detect = model(pic)
            detect_outputs.append(detect * w_det)
        sum_detect = torch.stack(detect_outputs, dim=0).sum(dim=0)
        avg_detect = sum_detect / sum(self.detection_weights)
        return avg_detect


    def train_ensemble(self, dataloaders): #csv_file, img_dir):
        mccs = []
        for i, md in enumerate(self.models_list):
            md.train_model(dataloaders[i])
            #torch.save(md.state_dict(), f"/home/team11/dev/MediSense/classification/t1/model_{laterality}_{view}.pth")

            metrics = md.evaluate_model(dataloaders[i])
            self.model_metrics.append(metrics)
            mccs.append(metrics['mcc'])
        
        self.detection_weights = _calculate_weights(mccs)

    def evaluate_ensemble(self, dataloaders):
        '''
        Evaluates following metrics of model: 
            precision, recall, F1 - per class
            MCC, F1 (multi-class macro average)
        Returns a dictionary with these metrics
        '''
   
        all_preds = []
        all_labels = []
        metrics = []
        
        for dataloader in dataloaders:
            inputslist = []
            labelslist = []
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputslist.append(inputs)
                labelslist.append(labels)

            # forward pass
            outputs = self.model(inputs)
            labels = labelslist[0] # should be the same across the list

            # get predictions
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # calculate precision, recall, f1-score per class
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
        
        # calculate multi-class macro average F1
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        # calculate Matthews Correlation Coefficient (MCC)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        metric = {
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'f1_macro': f1_macro,
            'mcc': mcc
        }
        
        return metrics

            
    def save_models(self, laterality, views, path):
        for i, model in enumerate(self.models_list):
            view = views[i]
            torch.save(model.state_dict(), f"{path}/model_{laterality}_{view}.pth")


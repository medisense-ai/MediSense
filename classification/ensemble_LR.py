import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.metrics import precision_recall_fscore_support, f1_score, matthews_corrcoef
import pickle
import numpy as np

def weighted_average(main_list, weights):
    """
    Function to multiply corresponding sublists with their weights and average them.

    Parameters:
    - main_list: List containing 2 lists of tensors.
    - weights: List of 2 weights.

    Returns:
    - A list of averaged tensors.
    """
    # Ensure that main_list contains exactly 2 sublists
    if len(main_list) != 2:
        raise ValueError("main_list must contain exactly 2 sublists")
    
    # Ensure that weights contains exactly 2 weights
    if len(weights) != 2:
        raise ValueError("weights must contain exactly 2 elements")
    
    # Extract sublists
    sublist1, sublist2 = main_list
    
    # Ensure that both sublists have the same length
    if len(sublist1) != len(sublist2):
        raise ValueError("Both sublists must have the same length")
    
    # Stack tensors from both sublists along a new dimension and apply weights
    stacked_tensors1 = np.array(sublist1) * weights[0]
    stacked_tensors2 = np.array(sublist2) * weights[1]
    
    # Sum the weighted tensors and average them
    sum_tensors = stacked_tensors1 + stacked_tensors2
    averaged_tensors = sum_tensors / sum(weights)
    
    # Return the averaged tensors as a list
    return averaged_tensors.tolist()

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
    def __init__(self, models : dict, detection_weights : dict = None):
        '''
        models: dictionary of models to ensemble, contains 'MLO' and 'CC' keys
        detection_weights: dictionary of ensemble weights for each model (not model weights)
        '''
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for k, model in models:
            model.to(device)

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
        
        print(f"Metrics: {self.model_metrics}")
        self.detection_weights = _calculate_weights(mccs)

        # save weights
        with open(f"/home/team11/dev/MediSense/classification/t1/ensemble_weights.pkl", 'wb') as f:
            pickle.dump(self.detection_weights, f)

    def evaluate_ensemble(self, dataloaders):
        '''
        Evaluates following metrics of model: 
            precision, recall, F1 - per class
            MCC, F1 (multi-class macro average)
        Returns a dictionary with these metrics
        '''

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        metrics = []
        
        preds = []
        labels = []
        for mdl, dataloader in zip(self.models_list, dataloaders):
            p, l = mdl.infer(dataloader)
            preds.append(p)
            labels.append(l)
            
        labels = labels[0] # should be the same across the list

        # get weighted predictions
        final_preds = torch.tensor(weighted_average(preds, self.detection_weights))
        print(final_preds.shape)
    
        # get predictions
        _, preds = torch.max(final_preds, 1)

        
        all_preds = preds.cpu().numpy()
        all_labels = labels.cpu().numpy()
        
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
        
        return metric

            
    def save_models(self, laterality, views, path):
        for i, model in enumerate(self.models_list):
            view = views[i]
            torch.save(model.state_dict(), f"{path}/model_{laterality}_{view}.pth")

        # Pickle and save the ensemble weights
        with open(f"{path}/ensemble_weights.pkl", 'wb') as f:
            pickle.dump(self.detection_weights, f)
                             

    def recalculate_weights(self, dataloaders=None, path=None):
        if dataloaders:
            mccs = []
            for i, md in enumerate(self.models_list):
                metrics = md.evaluate_model(dataloaders[i])
                self.model_metrics.append(metrics)
                mccs.append(metrics['mcc'])
            
            self.detection_weights = _calculate_weights(mccs)
        elif path:
            with open(f"{path}/ensemble_weights.pkl", 'rb') as f:
                self.detection_weights = pickle.load(f)
        else:
            raise ValueError("Either dataloaders or path should be provided.")

        

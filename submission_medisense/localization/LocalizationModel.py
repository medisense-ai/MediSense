import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from logger import Logger, log_cuda_memory

class MammoLocalizationResNet50(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, logger=None):
        """
        Initializes a localization model using a ResNet-50 backbone with a Feature Pyramid Network,
        and custom anchors for small lesions.
        
        Args:
            num_classes (int): Number of classes including background (i.e. background + 6 lesion types).
            pretrained (bool): Whether to use pretrained weights for the backbone.
            logger (optional): Logger for info messages.
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        
        # Create backbone using ResNet50 with FPN.
        backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)
        
        # Anchor generator tuned for smaller objects.
        anchor_generator = AnchorGenerator(
            sizes=tuple([(16, 32, 64, 128, 256)] * 5),
            aspect_ratios=tuple([(0.5, 1.0, 2.0)] * 5)
        )
        
        # Build the Faster R-CNN model using the backbone and custom anchor generator.
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            rpn_positive_fraction=0.7
        )
        self.model.to(self.device)

    def forward(self, images, targets=None):
        """
        Forward pass for the localization model.
        """
        processed_images = [img.to(self.device) for img in images]
        return self.model(processed_images, targets)

    def load_model(self, model_path):
        """
        Loads model weights from a file.
        """
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def train_model(self, dataloader, num_epochs=10, lr=0.005):
        """
        Trains the Faster R-CNN model using the provided dataloader.
        """
        if self.logger:
            self.logger.info("Starting training of the localization model")
        
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        self.model.to(self.device)
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            if self.logger:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                new_targets = []
                for t in targets:
                    if isinstance(t, dict):
                        new_targets.append({k: v.to(self.device) for k, v in t.items()})
                    else:
                        new_targets.append({
                            "boxes": torch.empty((0, 4), dtype=torch.float32).to(self.device),
                            "labels": torch.empty((0,), dtype=torch.int64).to(self.device)
                        })
                targets = new_targets
                
                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
            
            avg_loss = epoch_loss / len(dataloader)
            if self.logger:
                self.logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        if self.logger:
            self.logger.info("Training complete")
        else:
            print("Training complete")

    def infer(self, dataloader):
        """
        Runs inference with the detection model.
        """
        self.model.eval()
        all_detections = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                
                batch_detections = [{k: v.cpu() for k, v in out.items()} for out in outputs]
                all_detections.extend(batch_detections)
                
                if targets is not None:
                    batch_targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
                    all_targets.extend(batch_targets)
        
        return all_detections, all_targets

    @staticmethod
    def compute_iou(box1, box2, eps=1e-8):
        """
        Computes the Intersection over Union (IoU) between two bounding boxes.
        """
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        
        inter_width = max(0, xB - xA + 1)
        inter_height = max(0, yB - yA + 1)
        inter_area = inter_width * inter_height
        
        area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        
        union = area_box1 + area_box2 - inter_area + eps
        iou = inter_area / union
        return iou

    def evaluate_model_iou(self, dataloader):
        """
        Evaluates the model by computing the mean IoU over the dataset.
        """
        self.model.eval()
        all_ious = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                
                for i, output in enumerate(outputs):
                    if 'boxes' not in targets[i]:
                        continue
                    gt_boxes = targets[i]['boxes']
                    pred_boxes = output['boxes']
                    
                    for gt_box in gt_boxes:
                        best_iou = 0.0
                        gt_box_np = gt_box.cpu().numpy()
                        for pred_box in pred_boxes:
                            pred_box_np = pred_box.cpu().numpy()
                            iou_val = self.compute_iou(gt_box_np, pred_box_np)
                            best_iou = max(best_iou, iou_val)
                        all_ious.append(best_iou)
        
        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
        metrics = {'mean_iou': mean_iou}
        return metrics

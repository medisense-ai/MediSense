import sys
import os

# Add the directory two levels up to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torch import device
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision import transforms

from common.local.dataset import MammographyDataset
from common.local.logger import Logger
from classification.model_resnet import MammoClassificationResNet50
from classification.ensemble_LR import WeightedEnsembleModel

BATCH_SIZE = 32
WORKERS = 0

log = Logger(log_dir='/home/team11/dev/MediSense/classification/temp', log_file='ensemble2.log')
device = device("cuda:0" if is_available() else "cpu")
log.info(f"Eval on device: {device}")


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

ensembles = {}
dl_val = {}

for laterality in ["L", "R"]:
    models = []
    dataloaders_val = []

    for view in ["CC", "MLO"]:

        log.info(f"Loading data for {laterality} {view}")

        ds = MammographyDataset(
            csv_file="/home/team11/dev/MediSense/classification/temp/val_labels.csv",
            img_dir="/home/data/train/images",
            laterality=laterality,
            view=view,
            transform=transform,
        )
        dataloader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=False)
        dataloaders_val.append(dataloader)

        model = MammoClassificationResNet50(logger=log)

        log.info(f"Loading model for {laterality} {view}")
        model.load_model(f"/home/team11/dev/MediSense/classification/t1/model_{laterality}_{view}.pth")
        models.append(model)


    log.info("Evaluating ensemble on validation set")
        
    ensemble = WeightedEnsembleModel(models)
    # ensemble.load_models(laterality=laterality, views=["CC", "MLO"], path="/home/team11/dev/MediSense/classification/t1", dataloaders=dataloaders_val)
    
    log.info(f'Model count: {len(ensemble.models_list)}')
    log.info(f'Input count: {len(dataloaders_val)}')
    log.info(ensemble.evaluate_ensemble(dataloaders_val))


    log.info("Done")

    



    
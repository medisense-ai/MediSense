from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class MammographyDataset(Dataset):

	def __init__(self, csv_file, img_dir, laterality, view, transform=None):
		self.data = pd.read_csv(csv_file)
		self.img_dir = img_dir
		self.laterality = laterality
		self.view = view
		self.transform = transform

		self.unique_case_ids = self.data['case_id'].unique()
    	# self.case_id_mapping = {old_id : new_id for new_id, old_id in enumerate(unique_case_ids)}
		
	def __len__(self):
		return len(self.data[(self.data["laterality"] == self.laterality) & (self.data["view"] == self.view)])
	

	def __getitem__(self, idx):
		actual_idx = self.unique_case_ids[idx]
		row = self.data[(self.data["case_id"] == actual_idx) & (self.data["laterality"] == self.laterality) & (self.data["view"] == self.view)]
		if row.empty:
			raise ValueError(f"Case ID {actual_idx} not found for laterality {self.laterality} and view {self.view}")
		label = row["category"].values[0]
		img_file = f'{actual_idx}/{row["image_id"].values[0]}.jpg'
		img_path = os.path.join(self.img_dir, img_file)
		image = Image.open(img_path)
        
		if self.transform:
			image = self.transform(image)
        
		return image, label

class MammographyDataset2(Dataset):

	def __init__(self, csv_file, img_dir, laterality, view, transform=None):
		self.data = pd.read_csv(csv_file)
		self.img_dir = img_dir
		self.laterality = laterality
		self.view = view
		self.transform = transform

		self.unique_case_ids = self.data['case_id'].unique()
    	# self.case_id_mapping = {old_id : new_id for new_id, old_id in enumerate(unique_case_ids)}
		
	def __len__(self):
		return len(self.data[(self.data["laterality"] == self.laterality) & (self.data["view"] == self.view)])
	

	def __getitem__(self, idx):
		actual_idx = self.unique_case_ids[idx]
		row = self.data[(self.data["case_id"] == actual_idx) & (self.data["laterality"] == self.laterality) & (self.data["view"] == self.view)]
		if row.empty:
			raise ValueError(f"Case ID {actual_idx} not found for laterality {self.laterality} and view {self.view}")
		label = row["birads"].values[0]
		img_file = f'{actual_idx}/{row["image_id"].values[0]}.jpg'
		img_path = os.path.join(self.img_dir, img_file)
		image = Image.open(img_path)
        
		if self.transform:
			image = self.transform(image)
        
		return image, label
	
# Example usage
if __name__ == "__main__":
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	
	dataset = MammographyDataset(csv_file='data/classification.csv', csv_image_col="image_id", img_dir='data/images', transform=transform)
	dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
	
	# Example iteration
	for images, descriptions in dataloader:
		print(images.shape)  # Expected: (batch_size, 3, 224, 224)
		print(descriptions)  # List of descriptions
		break

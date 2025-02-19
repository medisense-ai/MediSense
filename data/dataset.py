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

	def __len__(self):
		return len(self.data[(self.data["laterality"] == self.laterality) & (self.data["view"] == self.view)])
	

	def __getitem__(self, idx):
		row = self.data[(self.data["case_id"] == idx) & (self.data["laterality"] == self.laterality) & (self.data["view"] == self.view)]
		print(idx)
		print(row)
		label = row["category"].values[0]
		img_file = f'{row["image_id"].values[0]}.jpg'
		img_path = os.path.join(self.img_dir, img_file)
		image = Image.open(img_path).convert("RGB")
        
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

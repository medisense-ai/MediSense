from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class MammographyDataset(Dataset):

	def __init__(self, csv_file, csv_image_col, img_dir, transform=None):
		self.data = pd.read_csv(csv_file)
		self.data_image_col = csv_image_col
		self.img_dir = img_dir
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, case_id):
		# retrieves a batch of (4) items
		img_list = []
		for ip in self.data[self.data["case_id"] == case_id]["image_id"]:
			x = Image.open(os.path.join(self.img_dir, f'{case_id}/{ip}'))
	
			if self.transform:
				x = self.transform(x)

			img_list.append(x)
        
		return img_list
	

	def get_image(self, idx):
		img_path = os.path.join(self.img_dir, self.data[self.data_image_col].iloc[idx])
		image = Image.open(img_path).convert("RGB")
        
		if self.transform:
			image = self.transform(image)
        
		return image

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

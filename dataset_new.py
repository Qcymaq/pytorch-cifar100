import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ChineseCharacterDataset(Dataset):
    def __init__(self, data_dir, char_to_label, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.char_to_label = char_to_label
        self.img_paths = []
        self.labels = []

        # Iterate through subdirectories and gather images and labels
        for label_folder in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_folder)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    self.img_paths.append(img_path)
                    self.labels.append(int(label_folder))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label



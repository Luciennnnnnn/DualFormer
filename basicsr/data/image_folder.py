from PIL import Image

from torch.utils.data import Dataset

class ImageFolder(Dataset):
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        image = Image.open(filepath).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
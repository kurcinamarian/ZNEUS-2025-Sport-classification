
import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SportsImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, split='train', augment=False):
        
        df = pd.read_csv(csv_path)
        
        self.df = df[df['data set'] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.augment = augment
        
        # Check for and remove invalid images
        valid_idx = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        
        for i, rel in enumerate(self.df["filepaths"]):
            p = os.path.join(self.root_dir, rel)
            if not os.path.exists(p) or os.path.splitext(p)[1].lower() not in valid_extensions:
                continue
            try:
                with Image.open(p) as image:
                    image.verify()
                valid_idx.append(i)
            except (UnidentifiedImageError, OSError):
                pass
            
        self.df = self.df.iloc[valid_idx].reset_index(drop=True)
        print(f"[{split}] kept {len(self.df)} samples after validation")
        
        if augment:
            # Perform image augmentations
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # Make sure all images are of size 224x224
                transforms.RandomHorizontalFlip(p=0.5), # 50% change to horizontally flip
                transforms.ColorJitter( # Randomly adjust color properties
                    brightness = 0.2,                # Adjust brightness
                    contrast = (0.8, 1.2),           # Adjust contrast
                    saturation = (0.8, 1.2),         # Adjust saturation
                    hue = 0.05                       # Adjust hue
                ),
                transforms.ToTensor(),
                transforms.Normalize( # Normalize RGB using ImageNet mean and std
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        # Get the image and label
        
        img_rel_path = self.df.iloc[idx]['filepaths']
        label = int(self.df.iloc[idx]['class id'])
        
        img_path = os.path.join(self.root_dir, img_rel_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
            
        return image, label
    

def create_dataloaders(csv_path, root_dir, batch_size=32, num_workers=4):
    
    # Create datasets for train, valid, test
    train_dataset = SportsImageDataset(csv_path, root_dir, split='train', augment=True)
    valid_dataset = SportsImageDataset(csv_path, root_dir, split='valid', augment=False)
    test_dataset = SportsImageDataset(csv_path, root_dir, split='test', augment=False)
    
    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader


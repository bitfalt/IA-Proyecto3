import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms


class ButterflyDataset(Dataset):
    """
    Custom Dataset for butterfly images with optional labels.
    Supports both labeled and unlabeled data for semi-supervised learning.
    """
    
    def __init__(self, root_dir, metadata_csv=None, transform=None, labeled=True):
        """
        Args:
            root_dir (str): Directory with all the images organized by class folders
            metadata_csv (str, optional): Path to CSV file with image metadata
            transform (callable, optional): Optional transform to be applied on images
            labeled (bool): Whether to return labels with images (for semi-supervised learning)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.labeled = labeled
        
        # Load metadata if provided
        if metadata_csv and os.path.exists(metadata_csv):
            self.metadata = pd.read_csv(metadata_csv)
            self.use_metadata = True
        else:
            self.use_metadata = False
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        # Get class names from directory structure
        if os.path.exists(root_dir):
            self.class_names = sorted([d for d in os.listdir(root_dir) 
                                     if os.path.isdir(os.path.join(root_dir, d))])
            
            # Create class to index mapping
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
            
            # Collect all image paths and labels
            for class_name in self.class_names:
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, img_name)
                            self.image_paths.append(img_path)
                            self.labels.append(self.class_to_idx[class_name])
        else:
            print(f"Warning: Directory {root_dir} does not exist")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return image with or without label
        if self.labeled:
            label = self.labels[idx]
            return image, label
        else:
            return image
    
    def get_class_names(self):
        """Return list of class names"""
        return self.class_names
    
    def get_num_classes(self):
        """Return number of classes"""
        return len(self.class_names)


def get_transforms(image_size=224, augment=True):
    """
    Get data transforms for training and validation.
    
    Args:
        image_size (int): Target image size
        augment (bool): Whether to apply data augmentation
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform 
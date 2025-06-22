import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from dataset import ButterflyDataset, get_transforms


class ButterflyDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for butterfly classification.
    Supports both supervised and semi-supervised learning scenarios.
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_csv: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        val_split: float = 0.2,
        test_split: float = 0.1,
        labeled_ratio: float = 0.3,  # For semi-supervised learning
        augment_data: bool = True,
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.metadata_csv = metadata_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        self.test_split = test_split
        self.labeled_ratio = labeled_ratio
        self.augment_data = augment_data
        self.seed = seed
        
        # Will be set during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.unlabeled_dataset = None
        self.labeled_dataset = None
        self.num_classes = None
        self.class_names = None
        
        # Set up transforms
        self.train_transform, self.val_transform = get_transforms(
            image_size=image_size, 
            augment=augment_data
        )
    
    def setup(self, stage=None):
        """
        Setup datasets for different stages.
        Creates train/val/test splits and handles semi-supervised learning splits.
        """
        # Load full dataset
        full_dataset = ButterflyDataset(
            root_dir=self.data_dir,
            metadata_csv=self.metadata_csv,
            transform=self.val_transform,  # Use val_transform for initial setup
            labeled=True
        )
        
        if len(full_dataset) == 0:
            raise ValueError(f"No data found in {self.data_dir}")
        
        self.num_classes = full_dataset.get_num_classes()
        self.class_names = full_dataset.get_class_names()
        
        print(f"Found {len(full_dataset)} images across {self.num_classes} classes")
        print(f"Classes: {self.class_names}")
        
        # Create stratified splits to maintain class distribution
        indices = list(range(len(full_dataset)))
        labels = [full_dataset.labels[i] for i in indices]
        
        # First split: train+val vs test
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_split,
            stratify=labels,
            random_state=self.seed
        )
        
        # Second split: train vs val
        train_val_labels = [labels[i] for i in train_val_indices]
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=self.val_split / (1 - self.test_split),  # Adjust for already removed test set
            stratify=train_val_labels,
            random_state=self.seed
        )
        
        print(f"Dataset splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # Create datasets with appropriate transforms
        if stage == "fit" or stage is None:
            # Training dataset with augmentation
            self.train_dataset = ButterflyDataset(
                root_dir=self.data_dir,
                metadata_csv=self.metadata_csv,
                transform=self.train_transform,
                labeled=True
            )
            self.train_dataset = Subset(self.train_dataset, train_indices)
            
            # Validation dataset without augmentation
            self.val_dataset = ButterflyDataset(
                root_dir=self.data_dir,
                metadata_csv=self.metadata_csv,
                transform=self.val_transform,
                labeled=True
            )
            self.val_dataset = Subset(self.val_dataset, val_indices)
            
            # For semi-supervised learning: split training data into labeled and unlabeled
            if self.labeled_ratio < 1.0:
                train_labels = [labels[i] for i in train_indices]
                labeled_indices, unlabeled_indices = train_test_split(
                    train_indices,
                    train_size=self.labeled_ratio,
                    stratify=train_labels,
                    random_state=self.seed
                )
                
                print(f"Semi-supervised split - Labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")
                
                # Labeled dataset
                self.labeled_dataset = ButterflyDataset(
                    root_dir=self.data_dir,
                    metadata_csv=self.metadata_csv,
                    transform=self.train_transform,
                    labeled=True
                )
                self.labeled_dataset = Subset(self.labeled_dataset, labeled_indices)
                
                # Unlabeled dataset (for autoencoder pretraining)
                self.unlabeled_dataset = ButterflyDataset(
                    root_dir=self.data_dir,
                    metadata_csv=self.metadata_csv,
                    transform=self.train_transform,
                    labeled=False  # Return only images, no labels
                )
                self.unlabeled_dataset = Subset(self.unlabeled_dataset, unlabeled_indices)
        
        if stage == "test" or stage is None:
            # Test dataset without augmentation
            self.test_dataset = ButterflyDataset(
                root_dir=self.data_dir,
                metadata_csv=self.metadata_csv,
                transform=self.val_transform,
                labeled=True
            )
            self.test_dataset = Subset(self.test_dataset, test_indices)
    
    def train_dataloader(self):
        """Return training dataloader"""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not set up. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not set up. Call setup() first.")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """Return test dataloader"""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not set up. Call setup() first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def labeled_dataloader(self):
        """Return labeled dataloader for semi-supervised learning"""
        if self.labeled_dataset is None:
            raise RuntimeError("Labeled dataset not set up. Call setup() with labeled_ratio < 1.0 first.")
        
        return DataLoader(
            self.labeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def unlabeled_dataloader(self):
        """Return unlabeled dataloader for autoencoder pretraining"""
        if self.unlabeled_dataset is None:
            raise RuntimeError("Unlabeled dataset not set up. Call setup() with labeled_ratio < 1.0 first.")
        
        return DataLoader(
            self.unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def get_dataset_info(self):
        """Return information about the dataset"""
        return {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'labeled_ratio': self.labeled_ratio,
            'train_size': len(self.train_dataset) if self.train_dataset else 0,
            'val_size': len(self.val_dataset) if self.val_dataset else 0,
            'test_size': len(self.test_dataset) if self.test_dataset else 0,
            'labeled_size': len(self.labeled_dataset) if self.labeled_dataset else 0,
            'unlabeled_size': len(self.unlabeled_dataset) if self.unlabeled_dataset else 0,
        } 
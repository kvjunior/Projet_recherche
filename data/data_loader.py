import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from distribution_tracker import DistributionTracker

class AdaptiveDataset(Dataset):
    """Dataset wrapper with distribution tracking capabilities"""
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.distribution_tracker = DistributionTracker()
        self.indices = list(range(len(dataset)))
        
        # Initialize distribution tracking
        self._analyze_distribution()
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple:
        real_idx = self.indices[idx]
        data, target = self.dataset[real_idx]
        
        if self.transform:
            data = self.transform(data)
            
        # Track access patterns
        self.distribution_tracker.update_access(real_idx, target)
        
        return data, target
    
    def _analyze_distribution(self):
        """Analyze data distribution"""
        targets = []
        for i in range(len(self.dataset)):
            _, target = self.dataset[i]
            targets.append(target)
        
        self.distribution_tracker.initialize_distribution(targets)
        
    def get_distribution_stats(self) -> Dict:
        """Get current distribution statistics"""
        return self.distribution_tracker.get_stats()
    
    def get_class_indices(self, class_idx: int) -> List[int]:
        """Get indices for a specific class"""
        return self.distribution_tracker.get_class_indices(class_idx)

class AdaptiveDataLoader:
    """Data loader with advanced partitioning capabilities"""
    def __init__(self, config: Dict):
        self.config = config
        self.transform_train = self._get_transforms(train=True)
        self.transform_test = self._get_transforms(train=False)
        
    def load_dataset(self, name: str, train: bool = True) -> AdaptiveDataset:
        """Load and wrap dataset with adaptive capabilities"""
        transform = self.transform_train if train else self.transform_test
        
        if name == 'mnist':
            dataset = datasets.MNIST(root='./data', train=train, download=True)
        elif name == 'cifar10':
            dataset = datasets.CIFAR10(root='./data', train=train, download=True)
        else:
            raise ValueError(f"Unknown dataset: {name}")
            
        return AdaptiveDataset(dataset, transform=transform)
    
    def create_client_partition(self, dataset: AdaptiveDataset,
                              num_clients: int,
                              partition_type: str = 'iid') -> List[DataLoader]:
        """Create data partition for federated learning clients"""
        if partition_type == 'iid':
            return self._create_iid_partition(dataset, num_clients)
        elif partition_type == 'dirichlet':
            return self._create_dirichlet_partition(dataset, num_clients)
        elif partition_type == 'shard':
            return self._create_shard_partition(dataset, num_clients)
        else:
            raise ValueError(f"Unknown partition type: {partition_type}")
    
    def _get_transforms(self, train: bool) -> transforms.Compose:
        """Get data transforms"""
        if self.config['dataset'] == 'mnist':
            if train:
                return transforms.Compose([
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
        elif self.config['dataset'] == 'cifar10':
            if train:
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010))
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010))
                ])
    
    def _create_iid_partition(self, dataset: AdaptiveDataset,
                            num_clients: int) -> List[DataLoader]:
        """Create IID data partition"""
        num_items = len(dataset)
        indices = list(range(num_items))
        np.random.shuffle(indices)
        
        # Split indices
        chunks = np.array_split(indices, num_clients)
        
        # Create dataloaders
        loaders = []
        for chunk in chunks:
            subset = Subset(dataset, chunk)
            loader = DataLoader(
                subset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers']
            )
            loaders.append(loader)
            
        return loaders
    
    def _create_dirichlet_partition(self, dataset: AdaptiveDataset,
                                  num_clients: int) -> List[DataLoader]:
        """Create non-IID partition using Dirichlet distribution"""
        num_classes = len(dataset.distribution_tracker.class_counts)
        client_data_indices = [[] for _ in range(num_clients)]
        
        # For each class
        for k in range(num_classes):
            class_indices = dataset.get_class_indices(k)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(
                np.repeat(self.config['dirichlet_alpha'], num_clients)
            )
            
            # Allocate indices according to proportions
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            splits = np.split(class_indices, proportions)
            
            # Assign splits to clients
            for client_id, indices in enumerate(splits):
                client_data_indices[client_id].extend(indices)
        
        # Create dataloaders
            loaders = []
            for indices in client_data_indices:
                subset = Subset(dataset, indices)
                loader = DataLoader(
                    subset,
                    batch_size=self.config['batch_size'],
                    shuffle=True,
                    num_workers=self.config['num_workers']
                )
                loaders.append(loader)
            
            return loaders
    
    def _create_shard_partition(self, dataset: AdaptiveDataset,
                              num_clients: int) -> List[DataLoader]:
        """Create non-IID partition using the shard method"""
        # Define number of shards (2 shards per client)
        num_shards = num_clients * 2
        num_imgs = len(dataset) // num_shards
        
        # Get all targets
        targets = []
        for idx in range(len(dataset)):
            _, target = dataset[idx]
            targets.append(target)
        targets = np.array(targets)
        
        # Sort indices by label
        idxs = np.argsort(targets)
        
        # Divide indices into shards
        idx_shards = np.array_split(idxs, num_shards)
        
        # Assign shards to clients
        client_data_indices = [[] for _ in range(num_clients)]
        shard_indices = list(range(num_shards))
        np.random.shuffle(shard_indices)
        
        for client_id in range(num_clients):
            # Assign 2 shards to each client
            shard1 = shard_indices[client_id * 2]
            shard2 = shard_indices[client_id * 2 + 1]
            client_data_indices[client_id].extend(idx_shards[shard1])
            client_data_indices[client_id].extend(idx_shards[shard2])
        
        # Create dataloaders
        loaders = []
        for indices in client_data_indices:
            subset = Subset(dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers']
            )
            loaders.append(loader)
            
        return loaders

    def get_validation_loader(self, dataset: AdaptiveDataset,
                            validation_split: float = 0.1) -> DataLoader:
        """Create validation dataloader"""
        num_samples = len(dataset)
        indices = list(range(num_samples))
        split = int(np.floor(validation_split * num_samples))
        
        np.random.shuffle(indices)
        val_indices = indices[:split]
        
        val_subset = Subset(dataset, val_indices)
        return DataLoader(
            val_subset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
    
    def create_meta_batch(self, dataset: AdaptiveDataset,
                         n_way: int,
                         k_shot: int,
                         n_query: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a meta-learning batch"""
        # Randomly select n_way classes
        classes = np.random.choice(
            list(dataset.distribution_tracker.class_counts.keys()),
            n_way,
            replace=False
        )
        
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        
        for i, cls in enumerate(classes):
            # Get indices for this class
            cls_indices = dataset.get_class_indices(cls)
            
            # Sample k_shot + n_query instances
            selected_indices = np.random.choice(
                cls_indices,
                k_shot + n_query,
                replace=False
            )
            
            # Split into support and query sets
            for idx in selected_indices[:k_shot]:
                x, _ = dataset[idx]
                support_xs.append(x)
                support_ys.append(i)
                
            for idx in selected_indices[k_shot:]:
                x, _ = dataset[idx]
                query_xs.append(x)
                query_ys.append(i)
        
        # Convert to tensors
        support_xs = torch.stack(support_xs)
        support_ys = torch.tensor(support_ys)
        query_xs = torch.stack(query_xs)
        query_ys = torch.tensor(query_ys)
        
        return support_xs, support_ys, query_xs, query_ys

    def analyze_partition_distribution(self, loaders: List[DataLoader]) -> Dict:
        """Analyze the distribution of data across clients"""
        distribution_stats = []
        
        for client_id, loader in enumerate(loaders):
            client_targets = []
            for _, targets in loader:
                client_targets.extend(targets.tolist())
            
            # Compute class distribution
            unique, counts = np.unique(client_targets, return_counts=True)
            dist = dict(zip(unique, counts))
            
            # Compute statistics
            stats = {
                'client_id': client_id,
                'total_samples': len(client_targets),
                'class_distribution': dist,
                'entropy': self._compute_distribution_entropy(counts)
            }
            distribution_stats.append(stats)
        
        return {
            'client_stats': distribution_stats,
            'num_clients': len(loaders),
            'total_dist_entropy': np.mean([s['entropy'] for s in distribution_stats])
        }
    
    def _compute_distribution_entropy(self, counts: np.ndarray) -> float:
        """Compute entropy of distribution"""
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def save_partition_info(self, loaders: List[DataLoader], path: str):
        """Save partition information"""
        partition_info = self.analyze_partition_distribution(loaders)
        torch.save(partition_info, path)
    
    def load_partition_info(self, path: str) -> Dict:
        """Load partition information"""
        return torch.load(path)
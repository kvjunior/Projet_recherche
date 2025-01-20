import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from config import AdaptiveHierFLConfig

class BaseTestCase(unittest.TestCase):
    """Base class for AdaptiveHierFL tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test config
        cls.config = AdaptiveHierFLConfig()
        cls.config.debug_mode = True
        cls.config.device = "cpu"  # Use CPU for testing
        
        # Set up test data directory
        cls.test_dir = Path(__file__).parent / "test_data"
        cls.test_dir.mkdir(exist_ok=True)
        
    def setUp(self):
        """Set up before each test"""
        pass
        
    def tearDown(self):
        """Clean up after each test"""
        # Clean test data directory
        if self.test_dir.exists():
            for file in self.test_dir.glob("*"):
                file.unlink()
                
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.test_dir.exists():
            cls.test_dir.rmdir()
            
    def create_dummy_model(self):
        """Create dummy model for testing"""
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 2)
                
            def forward(self, x):
                return self.fc(x)
        
        return DummyModel()
    
    def create_dummy_data(self, num_samples=100):
        """Create dummy data for testing"""
        x = torch.randn(num_samples, 10)
        y = torch.randint(0, 2, (num_samples,))
        return x, y
    
    def create_dummy_dataloader(self, num_samples=100, batch_size=32):
        """Create dummy dataloader for testing"""
        x, y = self.create_dummy_data(num_samples)
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    def assert_tensors_close(self, tensor1, tensor2, rtol=1e-5, atol=1e-8):
        """Assert that two tensors are close"""
        self.assertTrue(
            torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol),
            f"Tensors not close:\n{tensor1}\n{tensor2}"
        )
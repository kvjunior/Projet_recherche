import unittest
from test_base import BaseTestCase
from knowledge_distillation import CrossTierDistillation
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestCrossTierDistillation(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.distillation = CrossTierDistillation(self.config)
        
        # Create teacher and student models
        self.teacher_model = self.create_test_model()
        self.student_model = self.create_test_model()
        
        # Create test data
        self.test_loader = self.create_dummy_dataloader(
            num_samples=100,
            batch_size=32
        )
        
    def test_distill_cloud_to_edge(self):
        """Test knowledge distillation from cloud to edge"""
        # Initial predictions
        x, _ = next(iter(self.test_loader))
        initial_preds = self.student_model(x)
        
        # Perform distillation
        distilled_model = self.distillation.distill_cloud_to_edge(
            teacher_model=self.teacher_model,
            student_model=self.student_model
        )
        
        # Verify distillation
        final_preds = distilled_model(x)
        self.assertFalse(torch.allclose(initial_preds, final_preds))
        
        # Check if student predictions are closer to teacher
        with torch.no_grad():
            teacher_preds = self.teacher_model(x)
            initial_diff = F.mse_loss(initial_preds, teacher_preds)
            final_diff = F.mse_loss(final_preds, teacher_preds)
            self.assertLess(final_diff, initial_diff)
            
    def test_distill_edge_to_client(self):
        """Test knowledge distillation from edge to client"""
        # Perform distillation
        distilled_model = self.distillation.distill_edge_to_client(
            teacher_model=self.teacher_model,
            student_model=self.student_model,
            client_data=self.test_loader
        )
        
        # Verify model updates
        x, _ = next(iter(self.test_loader))
        with torch.no_grad():
            teacher_preds = self.teacher_model(x)
            student_preds = distilled_model(x)
            
            # Check if predictions are similar but not identical
            similarity = F.cosine_similarity(
                teacher_preds.mean(dim=0),
                student_preds.mean(dim=0),
                dim=0
            )
            self.assertGreater(similarity, 0.5)
            self.assertLess(similarity, 1.0)
            
    def test_distill_at_edge(self):
        """Test knowledge distillation at edge server"""
        # Perform distillation
        distilled_model = self.distillation.distill_at_edge(
            student_model=self.student_model,
            teacher_model=self.teacher_model
        )
        
        # Verify attention transfer
        x, _ = next(iter(self.test_loader))
        with torch.no_grad():
            _, teacher_attention = self.teacher_model.forward_with_attention(x)
            _, student_attention = distilled_model.forward_with_attention(x)
            
            # Check attention map similarity
            attention_similarity = F.cosine_similarity(
                teacher_attention.view(-1),
                student_attention.view(-1),
                dim=0
            )
            self.assertGreater(attention_similarity, 0.5)
            
    def test_adaptive_distillation(self):
        """Test adaptive knowledge distillation"""
        # Create validation loader
        val_loader = self.create_dummy_dataloader(
            num_samples=50,
            batch_size=16
        )
        
        # Perform adaptive distillation
        distilled_model = self.distillation.adaptive_distillation(
            student_model=self.student_model,
            teacher_model=self.teacher_model,
            validation_loader=val_loader
        )
        
        # Verify adaptation
        self.assertGreater(len(self.distillation.adaptation_history), 0)
        
    def test_temperature_scheduling(self):
        """Test temperature scheduling during distillation"""
        initial_temp = self.distillation.temperature
        
        # Update temperature multiple times
        for epoch in range(5):
            self.distillation.update_temperature(epoch)
            
        # Verify temperature changes
        self.assertNotEqual(initial_temp, self.distillation.temperature)
        
    def create_test_model(self):
        """Create test model with attention mechanism"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 20),
                    nn.ReLU()
                )
                self.attention = nn.Linear(20, 1)
                self.classifier = nn.Linear(20, 2)
                
            def forward(self, x):
                features = self.features(x)
                return self.classifier(features)
                
            def forward_with_attention(self, x):
                features = self.features(x)
                attention = torch.sigmoid(self.attention(features))
                output = self.classifier(features * attention)
                return output, attention
                
        return TestModel()
        
    def test_compute_distillation_loss(self):
        """Test computation of distillation loss"""
        batch_size = 32
        num_classes = 2
        temperature = 2.0
        
        # Create dummy logits
        student_logits = torch.randn(batch_size, num_classes)
        teacher_logits = torch.randn(batch_size, num_classes)
        
        # Compute soft probabilities
        student_soft = F.softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        
        # Compute distillation loss
        loss = self.distillation._compute_distillation_loss(
            student_soft=student_soft,
            teacher_soft=teacher_soft,
            temperature=temperature
        )
        
        # Verify loss properties
        self.assertGreater(loss.item(), 0)
        self.assertLess(loss.item(), float('inf'))
        
    def test_compute_attention_loss(self):
        """Test computation of attention transfer loss"""
        batch_size = 32
        feature_dim = 20
        
        # Create dummy attention maps
        student_attention = torch.randn(batch_size, 4, 5, feature_dim)
        teacher_attention = torch.randn(batch_size, 4, 5, feature_dim)
        
        # Compute attention loss
        loss = self.distillation._compute_attention_loss(
            student_attention=student_attention,
            teacher_attention=teacher_attention
        )
        
        # Verify loss properties
        self.assertGreater(loss.item(), 0)
        self.assertLess(loss.item(), float('inf'))
        
    def test_normalize_attention(self):
        """Test attention map normalization"""
        batch_size = 32
        feature_dim = 20
        
        # Create dummy attention map
        attention = torch.randn(batch_size, 4, 5, feature_dim)
        
        # Normalize attention
        normalized = self.distillation._normalize_attention(attention)
        
        # Verify normalization
        norms = torch.norm(normalized, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        
    def test_save_load_state(self):
        """Test saving and loading distillation state"""
        # Save state
        save_path = self.test_dir / "distillation_state.pt"
        self.distillation.save_state(save_path)
        
        # Modify state
        original_temp = self.distillation.temperature
        self.distillation.temperature = 5.0
        
        # Load state
        self.distillation.load_state(save_path)
        
        # Verify state restoration
        self.assertEqual(self.distillation.temperature, original_temp)
        
    def test_distillation_with_different_architectures(self):
        """Test distillation between different model architectures"""
        # Create models with different architectures
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
                
            def forward(self, x):
                return self.fc(x)
                
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 2)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                return self.fc2(x)
        
        # Create teacher (large) and student (small) models
        teacher = LargeModel()
        student = SmallModel()
        
        # Perform distillation
        distilled_model = self.distillation.distill_cloud_to_edge(
            teacher_model=teacher,
            student_model=student
        )
        
        # Verify model can make predictions
        x, _ = next(iter(self.test_loader))
        outputs = distilled_model(x)
        self.assertEqual(outputs.shape, (x.shape[0], 2))
        
    def test_distillation_with_noisy_data(self):
        """Test distillation robustness with noisy data"""
        # Create noisy data loader
        x, y = self.create_dummy_data(num_samples=100)
        noise = torch.randn_like(x) * 0.1
        noisy_x = x + noise
        
        noisy_dataset = torch.utils.data.TensorDataset(noisy_x, y)
        noisy_loader = torch.utils.data.DataLoader(
            noisy_dataset,
            batch_size=32
        )
        
        # Perform distillation with noisy data
        distilled_model = self.distillation.distill_edge_to_client(
            teacher_model=self.teacher_model,
            student_model=self.student_model,
            client_data=noisy_loader
        )
        
        # Verify model still works on clean data
        clean_x, _ = next(iter(self.test_loader))
        outputs = distilled_model(clean_x)
        self.assertEqual(outputs.shape, (clean_x.shape[0], 2))
        
    def test_batch_size_sensitivity(self):
        """Test distillation with different batch sizes"""
        batch_sizes = [1, 4, 16, 64]
        
        for batch_size in batch_sizes:
            # Create data loader with current batch size
            loader = self.create_dummy_dataloader(
                num_samples=100,
                batch_size=batch_size
            )
            
            # Perform distillation
            distilled_model = self.distillation.distill_edge_to_client(
                teacher_model=self.teacher_model,
                student_model=self.student_model,
                client_data=loader
            )
            
            # Verify model works
            x, _ = next(iter(loader))
            outputs = distilled_model(x)
            self.assertEqual(outputs.shape, (x.shape[0], 2))

if __name__ == '__main__':
    unittest.main()
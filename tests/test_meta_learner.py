import unittest
from test_base import BaseTestCase
from meta_learner import MetaLearner, TaskBatch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TestMetaLearner(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.meta_learner = MetaLearner(self.config)
        
        # Create model and data
        self.model = self.create_meta_test_model()
        self.test_data = self.create_meta_test_data()
        
    def test_generate_tasks(self):
        """Test meta-learning task generation"""
        client = self.create_dummy_client()
        tasks = self.meta_learner.generate_tasks(client)
        
        # Verify task properties
        self.assertEqual(len(tasks), self.config.meta_learning.meta_batch_size)
        
        for task in tasks:
            # Check support set
            self.assertIsInstance(task, TaskBatch)
            self.assertEqual(len(task.support_data), self.config.meta_learning.k_shot * self.config.meta_learning.n_way)
            
            # Check query set
            self.assertEqual(len(task.query_data), self.config.meta_learning.n_query * self.config.meta_learning.n_way)
            
            # Verify labels are properly distributed
            support_labels = torch.unique(task.support_labels)
            query_labels = torch.unique(task.query_labels)
            self.assertEqual(len(support_labels), self.config.meta_learning.n_way)
            self.assertEqual(len(query_labels), self.config.meta_learning.n_way)
            
    def test_meta_train(self):
        """Test meta-training process"""
        # Create tasks
        tasks = [self.create_dummy_task() for _ in range(self.config.meta_learning.meta_batch_size)]
        
        # Initial predictions
        x, _ = self.test_data
        initial_preds = self.model(x)
        
        # Perform meta-training
        trained_model = self.meta_learner.meta_train(
            model=self.model,
            tasks=tasks
        )
        
        # Verify model updates
        final_preds = trained_model(x)
        self.assertFalse(torch.allclose(initial_preds, final_preds))
        
        # Check meta-learning properties
        meta_test_accuracy = self._evaluate_meta_learning(trained_model, tasks)
        self.assertGreater(meta_test_accuracy, 0.2)  # Better than random
        
    def test_personalization(self):
        """Test model personalization for specific client"""
        client_data = self.create_dummy_dataloader()
        
        # Initial predictions
        x, _ = next(iter(client_data))
        initial_preds = self.model(x)
        
        # Perform personalization
        personalized_model = self.meta_learner.personalize(
            model=self.model,
            client_data=client_data
        )
        
        # Verify personalization
        final_preds = personalized_model(x)
        self.assertFalse(torch.allclose(initial_preds, final_preds))
        
        # Check adaptation to client data
        accuracy = self._evaluate_model(personalized_model, client_data)
        self.assertGreater(accuracy, 0.2)  # Better than random
        
    def test_prototype_learning(self):
        """Test prototype-based meta-learning"""
        support_data = torch.randn(10, 10)
        support_labels = torch.randint(0, 2, (10,))
        
        # Compute prototypes
        prototypes = self.meta_learner._compute_prototypes(
            support_data=support_data,
            support_labels=support_labels,
            model=self.model
        )
        
        # Verify prototype properties
        num_classes = len(torch.unique(support_labels))
        self.assertEqual(len(prototypes), num_classes)
        
        # Test prototype classification
        query_data = torch.randn(5, 10)
        query_embeddings = self.model.get_embeddings(query_data)
        
        loss = self.meta_learner._compute_prototype_loss(
            query_embeddings=query_embeddings,
            query_labels=torch.randint(0, 2, (5,)),
            prototypes=prototypes
        )
        
        self.assertGreater(loss.item(), 0)
        
    def test_confidence_weighting(self):
        """Test confidence-based weighting in meta-learning"""
        logits = torch.randn(10, 2)
        weights = self.meta_learner._compute_confidence_weights(logits)
        
        # Verify weight properties
        self.assertEqual(len(weights), len(logits))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights <= 1))
        
    def test_task_memory(self):
        """Test episodic memory for meta-learning tasks"""
        # Initialize memory
        self.meta_learner._initialize_task_memory()
        
        # Create and add tasks
        task = self.create_dummy_task()
        performance = 0.75
        
        self.meta_learner._update_task_memory(task, performance)
        
        # Verify memory update
        self.assertEqual(len(self.meta_learner.task_memory['support_data']), 1)
        self.assertEqual(len(self.meta_learner.task_memory['performance']), 1)
        
        # Sample from memory
        sampled_task = self.meta_learner._sample_memory_task()
        self.assertIsInstance(sampled_task, TaskBatch)
        
    def create_meta_test_model(self):
        """Create model for meta-learning tests"""
        class MetaTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 20)
                )
                self.classifier = nn.Linear(20, 2)
                
            def forward(self, x):
                features = self.encoder(x)
                return self.classifier(features)
                
            def get_embeddings(self, x):
                return self.encoder(x)
                
        return MetaTestModel()
        
    def create_meta_test_data(self):
        """Create data for meta-learning tests"""
        x = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        return x, y
        
    def create_dummy_task(self):
        """Create dummy task for testing"""
        support_data = torch.randn(
            self.config.meta_learning.k_shot * self.config.meta_learning.n_way,
            10
        )
        support_labels = torch.randint(
            0,
            self.config.meta_learning.n_way,
            (self.config.meta_learning.k_shot * self.config.meta_learning.n_way,)
        )
        
        query_data = torch.randn(
            self.config.meta_learning.n_query * self.config.meta_learning.n_way,
            10
        )
        query_labels = torch.randint(
            0,
            self.config.meta_learning.n_way,
            (self.config.meta_learning.n_query * self.config.meta_learning.n_way,)
        )
        
        return TaskBatch(
            support_data=support_data,
            support_labels=support_labels,
            query_data=query_data,
            query_labels=query_labels
        )
        
    def create_dummy_client(self):
        """Create dummy client for testing"""
        return type('DummyClient', (), {
            'id': 'test_client',
            'train_loader': self.create_dummy_dataloader(),
            'test_loader': self.create_dummy_dataloader(),
            'model': self.create_dummy_model()
        })
        
    def _evaluate_meta_learning(self, model, tasks):
        """Evaluate meta-learning performance"""
        correct = 0
        total = 0
        
        for task in tasks:
            with torch.no_grad():
                outputs = model(task.query_data)
                _, predicted = torch.max(outputs.data, 1)
                total += task.query_labels.size(0)
                correct += (predicted == task.query_labels).sum().item()
                
        return correct / total
        
    def _evaluate_model(self, model, dataloader):
        """Evaluate model performance"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in dataloader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return correct / total
        
    def test_meta_batch_size_effects(self):
        """Test effects of different meta batch sizes"""
        original_batch_size = self.config.meta_learning.meta_batch_size
        test_batch_sizes = [1, 2, 4, 8]
        
        for batch_size in test_batch_sizes:
            self.config.meta_learning.meta_batch_size = batch_size
            tasks = [self.create_dummy_task() for _ in range(batch_size)]
            
            # Meta-train with current batch size
            trained_model = self.meta_learner.meta_train(
                model=self.create_meta_test_model(),
                tasks=tasks
            )
            
            # Verify training completes successfully
            self.assertIsNotNone(trained_model)
            
        # Restore original batch size
        self.config.meta_learning.meta_batch_size = original_batch_size
        
    def test_learning_rate_adaptation(self):
        """Test adaptation of meta-learning rate"""
        initial_lr = self.meta_learner.meta_lr
        
        # Train for multiple rounds
        for _ in range(5):
            tasks = [self.create_dummy_task() for _ in range(self.config.meta_learning.meta_batch_size)]
            self.meta_learner.meta_train(
                model=self.model,
                tasks=tasks
            )
            
        # Verify learning rate adaptation
        self.assertNotEqual(self.meta_learner.meta_lr, initial_lr)
        
    def test_save_load_meta_state(self):
        """Test saving and loading meta-learner state"""
        # Save state
        save_path = self.test_dir / "meta_state.pt"
        self.meta_learner.save_meta_state(save_path)
        
        # Modify state
        original_lr = self.meta_learner.meta_lr
        self.meta_learner.meta_lr = 0.1
        
        # Load state
        self.meta_learner.load_meta_state(save_path)
        
        # Verify state restoration
        self.assertEqual(self.meta_learner.meta_lr, original_lr)
        
    def test_meta_learning_with_different_architectures(self):
        """Test meta-learning with different model architectures"""
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(10, 2)
                
            def forward(self, x):
                return self.net(x)
                
            def get_embeddings(self, x):
                return x
        
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(10, 50),
                    nn.ReLU(),
                    nn.Linear(50, 2)
                )
                
            def forward(self, x):
                return self.net(x)
                
            def get_embeddings(self, x):
                return self.net[:-1](x)
        
        models = [SmallModel(), LargeModel()]
        
        for model in models:
            tasks = [self.create_dummy_task() 
                    for _ in range(self.config.meta_learning.meta_batch_size)]
            
            # Test meta-training
            trained_model = self.meta_learner.meta_train(
                model=model,
                tasks=tasks
            )
            
            # Verify model updates
            self.assertIsNotNone(trained_model)
            
    def test_robustness_to_noisy_tasks(self):
        """Test meta-learning robustness with noisy tasks"""
        # Create tasks with noise
        noisy_tasks = []
        for _ in range(self.config.meta_learning.meta_batch_size):
            task = self.create_dummy_task()
            # Add noise to support and query data
            noise_level = 0.1
            task.support_data += torch.randn_like(task.support_data) * noise_level
            task.query_data += torch.randn_like(task.query_data) * noise_level
            noisy_tasks.append(task)
            
        # Train with noisy tasks
        trained_model = self.meta_learner.meta_train(
            model=self.model,
            tasks=noisy_tasks
        )
        
        # Evaluate on clean task
        clean_task = self.create_dummy_task()
        with torch.no_grad():
            outputs = trained_model(clean_task.query_data)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == clean_task.query_labels).float().mean()
        
        # Verify model still performs reasonably
        self.assertGreater(accuracy, 0.2)  # Better than random
        
    def test_meta_learning_convergence(self):
        """Test convergence of meta-learning process"""
        # Train for multiple epochs
        losses = []
        for _ in range(5):
            tasks = [self.create_dummy_task() 
                    for _ in range(self.config.meta_learning.meta_batch_size)]
            
            # Meta-train and record loss
            _, meta_loss = self.meta_learner.meta_train(
                model=self.model,
                tasks=tasks,
                return_loss=True
            )
            losses.append(meta_loss)
            
        # Verify loss decreases
        self.assertLess(losses[-1], losses[0])

if __name__ == '__main__':
    unittest.main()
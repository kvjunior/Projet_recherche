import unittest
from test_base import BaseTestCase
from dynamic_tier_manager import DynamicTierManager
import numpy as np
import torch

class TestDynamicTierManager(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.tier_manager = DynamicTierManager(self.config)
        
    def test_form_initial_tiers(self):
        """Test initial tier formation"""
        # Create dummy clients and edges
        num_clients = 10
        num_edges = 3
        
        clients = [self.create_dummy_client(i) for i in range(num_clients)]
        edges = [self.create_dummy_edge(i) for i in range(num_edges)]
        cloud = self.create_dummy_cloud()
        
        # Form initial tiers
        tier_structure = self.tier_manager.form_initial_tiers(
            clients=clients,
            edges=edges,
            cloud=cloud
        )
        
        # Verify structure
        self.assertIn('num_tiers', tier_structure)
        self.assertIn('tier_assignments', tier_structure)
        self.assertIn('tier_connections', tier_structure)
        
        # Check number of tiers
        self.assertGreaterEqual(tier_structure['num_tiers'], self.config.tier.min_tiers)
        self.assertLessEqual(tier_structure['num_tiers'], self.config.tier.max_tiers)
        
        # Check assignments
        assignments = tier_structure['tier_assignments']
        self.assertEqual(len(assignments), num_clients + num_edges + 1)  # +1 for cloud
        
        # Verify connections
        connections = tier_structure['tier_connections']
        self.assertGreater(len(connections), 0)
        
    def test_adapt_tiers(self):
        """Test tier adaptation"""
        # Create initial structure
        initial_structure = {
            'num_tiers': 3,
            'tier_assignments': {
                'client_0': 0,
                'client_1': 0,
                'edge_0': 1,
                'cloud': 2
            },
            'tier_connections': ['client_0', 'edge_0']
        }
        
        # Create metrics
        metrics = {
            'communication_cost': [100, 90, 80, 85, 95],
            'test_accuracy': [0.6, 0.65, 0.7, 0.68, 0.66]
        }
        
        # Adapt tiers
        new_structure = self.tier_manager.adapt_tiers(
            current_structure=initial_structure,
            metrics=metrics
        )
        
        # Verify adaptation
        self.assertIsNotNone(new_structure)
        self.assertIn('num_tiers', new_structure)
        self.assertIn('tier_assignments', new_structure)
        self.assertIn('tier_connections', new_structure)
        
    def test_analyze_clients(self):
        """Test client analysis"""
        num_clients = 5
        clients = [self.create_dummy_client(i) for i in range(num_clients)]
        
        client_metrics = self.tier_manager._analyze_clients(clients)
        
        self.assertEqual(len(client_metrics), num_clients)
        for metrics in client_metrics.values():
            self.assertIn('compute_capacity', metrics)
            self.assertIn('data_distribution', metrics)
            self.assertIn('network_reliability', metrics)
            
    def test_cluster_nodes(self):
        """Test node clustering"""
        # Create test data
        num_nodes = 10
        client_metrics = {
            i: {
                'compute_capacity': np.random.random(),
                'network_reliability': np.random.random(),
                'data_distribution': np.random.random(5)
            }
            for i in range(num_nodes)
        }
        
        network_conditions = {
            'latency_matrix': np.random.random((num_nodes, num_nodes)),
            'bandwidth_matrix': np.random.random((num_nodes, num_nodes))
        }
        
        # Perform clustering
        tier_assignments = self.tier_manager._cluster_nodes(
            clients=[],  # Dummy clients
            edges=[],    # Dummy edges
            client_metrics=client_metrics,
            network_conditions=network_conditions
        )
        
        # Verify clustering results
        self.assertEqual(len(tier_assignments), num_nodes)
        unique_tiers = len(set(tier_assignments.values()))
        self.assertGreaterEqual(unique_tiers, self.config.tier.min_tiers)
        self.assertLessEqual(unique_tiers, self.config.tier.max_tiers)
        
    def create_dummy_client(self, client_id):
        """Create dummy client for testing"""
        return type('DummyClient', (), {
            'id': f'client_{client_id}',
            'train_loader': self.create_dummy_dataloader(),
            'test_loader': self.create_dummy_dataloader(),
            'model': self.create_dummy_model()
        })
        
    def create_dummy_edge(self, edge_id):
        """Create dummy edge for testing"""
        return type('DummyEdge', (), {
            'id': f'edge_{edge_id}',
            'model': self.create_dummy_model(),
            'client_ids': [f'client_{i}' for i in range(3)]
        })
        
    def create_dummy_cloud(self):
        """Create dummy cloud for testing"""
        return type('DummyCloud', (), {
            'id': 'cloud',
            'model': self.create_dummy_model()
        })
        
if __name__ == '__main__':
    unittest.main()
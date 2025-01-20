import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
import logging

from dynamic_tier_manager import DynamicTierManager
from knowledge_distillation import CrossTierDistillation
from meta_learner import MetaLearner
from adaptive_client import AdaptiveClient
from adaptive_edge import AdaptiveEdge 
from adaptive_cloud import AdaptiveCloud
from utils.metrics import compute_metrics
from utils.communication import Communication
from config import AdaptiveHierFLConfig

class AdaptiveHierFL:
    def __init__(self, config: AdaptiveHierFLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize core components
        self.tier_manager = DynamicTierManager(config)
        self.distillation = CrossTierDistillation(config)
        self.meta_learner = MetaLearner(config)
        self.comm = Communication(config)
        
        # Initialize metrics tracking
        self.metrics = {
            'train_loss': [],
            'test_accuracy': [],
            'communication_cost': [],
            'tier_adaptations': []
        }
        
        self.logger = logging.getLogger(__name__)

    def initialize_network(self, clients: List[AdaptiveClient], 
                         initial_edges: List[AdaptiveEdge],
                         cloud: AdaptiveCloud):
        """Initialize the hierarchical network structure"""
        self.clients = clients
        self.edges = initial_edges
        self.cloud = cloud
        
        # Initial tier formation
        self.tier_structure = self.tier_manager.form_initial_tiers(
            clients=clients,
            edges=initial_edges,
            cloud=cloud
        )
        
    def train(self, num_rounds: int):
        """Main training loop"""
        for round_idx in tqdm(range(num_rounds)):
            # 1. Check and adapt tier structure if needed
            if round_idx % self.config.tier_adaptation_frequency == 0:
                self.tier_structure = self.tier_manager.adapt_tiers(
                    current_structure=self.tier_structure,
                    metrics=self.metrics
                )
            
            # 2. Client training with meta-learning
            client_updates = self._train_clients()
            
            # 3. Edge aggregation with knowledge distillation
            edge_updates = self._aggregate_at_edges(client_updates)
            
            # 4. Cloud aggregation and global model update
            global_model = self._aggregate_at_cloud(edge_updates)
            
            # 5. Cross-tier knowledge transfer
            self._perform_knowledge_transfer()
            
            # 6. Update metrics
            self._update_metrics(round_idx)
            
            # 7. Log progress
            self._log_round_info(round_idx)

    def _train_clients(self) -> Dict[int, Any]:
        """Train clients using meta-learning"""
        client_updates = {}
        for client in self.clients:
            # Get personalized meta-learning tasks
            tasks = self.meta_learner.generate_tasks(client)
            
            # Perform meta-training
            updated_model = self.meta_learner.meta_train(
                model=client.model,
                tasks=tasks
            )
            
            client_updates[client.id] = {
                'model': updated_model,
                'metadata': client.get_training_metadata()
            }
        
        return client_updates

    def _aggregate_at_edges(self, client_updates: Dict) -> Dict[int, Any]:
        """Aggregate client updates at edge servers"""
        edge_updates = {}
        for edge in self.edges:
            # Collect relevant client updates
            edge_client_updates = {
                cid: client_updates[cid] 
                for cid in edge.client_ids
            }
            
            # Perform edge-level aggregation
            aggregated_model = edge.aggregate(edge_client_updates)
            
            # Apply knowledge distillation if needed
            if self.config.use_distillation:
                aggregated_model = self.distillation.distill_at_edge(
                    student_model=aggregated_model,
                    teacher_model=self.cloud.model
                )
            
            edge_updates[edge.id] = {
                'model': aggregated_model,
                'metadata': edge.get_aggregation_metadata()
            }
            
        return edge_updates

    def _aggregate_at_cloud(self, edge_updates: Dict) -> torch.nn.Module:
        """Aggregate edge updates at cloud server"""
        return self.cloud.aggregate(edge_updates)

    def _perform_knowledge_transfer(self):
        """Perform cross-tier knowledge transfer"""
        # Cloud to edge knowledge transfer
        for edge in self.edges:
            self.distillation.distill_cloud_to_edge(
                teacher_model=self.cloud.model,
                student_model=edge.model
            )
        
        # Edge to client knowledge transfer
        for edge in self.edges:
            for client_id in edge.client_ids:
                client = self.clients[client_id]
                self.distillation.distill_edge_to_client(
                    teacher_model=edge.model,
                    student_model=client.model
                )

    def _update_metrics(self, round_idx: int):
        """Update training metrics"""
        metrics = compute_metrics(
            clients=self.clients,
            edges=self.edges,
            cloud=self.cloud,
            round_idx=round_idx
        )
        
        for key, value in metrics.items():
            self.metrics[key].append(value)

    def _log_round_info(self, round_idx: int):
        """Log training progress"""
        self.logger.info(f"Round {round_idx} completed:")
        self.logger.info(f"Train Loss: {self.metrics['train_loss'][-1]:.4f}")
        self.logger.info(f"Test Accuracy: {self.metrics['test_accuracy'][-1]:.4f}")
        self.logger.info(f"Communication Cost: {self.metrics['communication_cost'][-1]}")
        
        if round_idx % self.config.tier_adaptation_frequency == 0:
            self.logger.info("Tier adaptation performed")
            self.logger.info(f"Current tier structure: {self.tier_structure}")
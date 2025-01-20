import torch
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import time
import queue
import threading

class CommunicationManager:
    """Manage communication between different tiers"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.message_queues = defaultdict(queue.Queue)
        self.bandwidth_tracker = BandwidthTracker()
        self.latency_tracker = LatencyTracker()
        self.compression_handler = CompressionHandler(config)
        
        # Communication statistics
        self.stats = defaultdict(list)
        self.active_transfers = {}
        
        # Initialize background monitoring
        self._start_monitoring()
        
    def send(self, source: str,
             destination: str,
             data: Any,
             priority: int = 0) -> str:
        """Send data between nodes"""
        # Generate transfer ID
        transfer_id = f"{source}_{destination}_{time.time()}"
        
        # Compress data if needed
        compressed_data = self.compression_handler.compress(data)
        
        # Create message
        message = {
            'id': transfer_id,
            'source': source,
            'destination': destination,
            'data': compressed_data,
            'priority': priority,
            'timestamp': time.time(),
            'size': self._get_data_size(compressed_data)
        }
        
        # Add to queue
        self.message_queues[destination].put((priority, message))
        
        # Track active transfer
        self.active_transfers[transfer_id] = message
        
        # Update statistics
        self._update_stats(message)
        
        return transfer_id
        
    def receive(self, destination: str,
                timeout: Optional[float] = None) -> Optional[Dict]:
        """Receive next message for destination"""
        try:
            priority, message = self.message_queues[destination].get(
                timeout=timeout
            )
            
            # Decompress data
            message['data'] = self.compression_handler.decompress(message['data'])
            
            # Update statistics
            self._update_stats(message, received=True)
            
            # Remove from active transfers
            transfer_id = message['id']
            if transfer_id in self.active_transfers:
                del self.active_transfers[transfer_id]
                
            return message
        except queue.Empty:
            return None
            
    def broadcast(self, source: str,
                 destinations: List[str],
                 data: Any,
                 priority: int = 0) -> List[str]:
        """Broadcast data to multiple destinations"""
        transfer_ids = []
        
        # Check if data can be shared
        shared_data = self.compression_handler.compress(data)
        
        for destination in destinations:
            transfer_id = self.send(
                source=source,
                destination=destination,
                data=shared_data,
                priority=priority
            )
            transfer_ids.append(transfer_id)
            
        return transfer_ids
        
    def check_status(self, transfer_id: str) -> Dict[str, Any]:
        """Check status of transfer"""
        if transfer_id not in self.active_transfers:
            return {'status': 'completed'}
            
        message = self.active_transfers[transfer_id]
        queue_position = self._get_queue_position(message)
        
        return {
            'status': 'pending',
            'queue_position': queue_position,
            'time_in_queue': time.time() - message['timestamp'],
            'estimated_time': self._estimate_transfer_time(message)
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            'bandwidth': self.bandwidth_tracker.get_stats(),
            'latency': self.latency_tracker.get_stats(),
            'compression_ratio': self.compression_handler.get_stats(),
            'queue_lengths': {
                dest: queue.qsize()
                for dest, queue in self.message_queues.items()
            }
        }
        
    def _start_monitoring(self):
        """Start background monitoring"""
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while True:
            # Update bandwidth measurements
            self.bandwidth_tracker.update()
            
            # Update latency measurements
            self.latency_tracker.update()
            
            # Clean up old transfers
            self._cleanup_old_transfers()
            
            # Sleep
            time.sleep(self.config.get('monitor_interval', 1.0))
            
    def _cleanup_old_transfers(self):
        """Clean up old completed transfers"""
        current_time = time.time()
        timeout = self.config.get('transfer_timeout', 300)  # 5 minutes
        
        to_remove = []
        for transfer_id, message in self.active_transfers.items():
            if current_time - message['timestamp'] > timeout:
                to_remove.append(transfer_id)
                
        for transfer_id in to_remove:
            del self.active_transfers[transfer_id]
            
    def _update_stats(self, message: Dict[str, Any],
                     received: bool = False):
        """Update communication statistics"""
        stats_key = f"{message['source']}_{message['destination']}"
        
        self.stats[f'{stats_key}_size'].append(message['size'])
        self.stats[f'{stats_key}_time'].append(time.time())
        
        if received:
            # Update latency tracking
            self.latency_tracker.add_measurement(
                source=message['source'],
                destination=message['destination'],
                latency=time.time() - message['timestamp']
            )
            
        # Update bandwidth tracking
        self.bandwidth_tracker.add_transfer(message['size'])
        
    def _get_queue_position(self, message: Dict[str, Any]) -> int:
        """Get position in destination queue"""
        queue = self.message_queues[message['destination']]
        position = 1
        
        for priority, queued_message in queue.queue:
            if priority < message['priority']:
                position += 1
                
        return position
        
    def _estimate_transfer_time(self, message: Dict[str, Any]) -> float:
        """Estimate transfer time based on current conditions"""
        # Get current bandwidth
        bandwidth = self.bandwidth_tracker.get_current_bandwidth()
        
        # Get current latency
        latency = self.latency_tracker.get_current_latency(
            message['source'],
            message['destination']
        )
        
        # Estimate time based on message size
        transfer_time = message['size'] / max(bandwidth, 1e-6)
        
        # Add latency and queue delay
        queue_delay = self._get_queue_position(message) * \
                     self.config.get('avg_transfer_time', 1.0)
                     
        return transfer_time + latency + queue_delay
        
    def _get_data_size(self, data: Any) -> int:
        """Get size of data in bytes"""
        if isinstance(data, torch.Tensor):
            return data.element_size() * data.nelement()
        elif isinstance(data, dict):
            return sum(self._get_data_size(v) for v in data.values())
        elif isinstance(data, (list, tuple)):
            return sum(self._get_data_size(v) for v in data)
        else:
            return len(str(data).encode())

class BandwidthTracker:
    """Track available bandwidth"""
    def __init__(self):
        self.transfer_history = []
        self.current_bandwidth = 0.0
        
    def add_transfer(self, size: int):
        """Add completed transfer"""
        self.transfer_history.append({
            'size': size,
            'timestamp': time.time()
        })
        
    def update(self):
        """Update bandwidth measurement"""
        if len(self.transfer_history) < 2:
            return
            
        # Clean old history
        current_time = time.time()
        self.transfer_history = [
            t for t in self.transfer_history
            if current_time - t['timestamp'] < 60  # Keep last minute
        ]
        
        if len(self.transfer_history) < 2:
            return
            
        # Calculate bandwidth over window
        total_size = sum(t['size'] for t in self.transfer_history)
        time_window = self.transfer_history[-1]['timestamp'] - \
                     self.transfer_history[0]['timestamp']
                     
        if time_window > 0:
            self.current_bandwidth = total_size / time_window
            
    def get_current_bandwidth(self) -> float:
        """Get current bandwidth measurement"""
        return self.current_bandwidth
        
    def get_stats(self) -> Dict[str, float]:
        """Get bandwidth statistics"""
        if not self.transfer_history:
            return {
                'current': 0.0,
                'average': 0.0,
                'peak': 0.0
            }
            
        current = self.current_bandwidth
        total_size = sum(t['size'] for t in self.transfer_history)
        total_time = (self.transfer_history[-1]['timestamp'] - 
                     self.transfer_history[0]['timestamp'])
        average = total_size / max(total_time, 1e-6)
        
        # Calculate peak bandwidth over 5-second windows
        peak = 0.0
        if len(self.transfer_history) > 1:
            window_size = 5.0
            for i in range(len(self.transfer_history)-1):
                window_start = self.transfer_history[i]['timestamp']
                window_end = window_start + window_size
                
                # Sum transfers in window
                window_size = sum(
                    t['size'] for t in self.transfer_history[i:]
                    if t['timestamp'] <= window_end
                )
                
                window_bandwidth = window_size / window_size
                peak = max(peak, window_bandwidth)
                
        return {
            'current': current,
            'average': average,
            'peak': peak
        }

class LatencyTracker:
    """Track communication latencies"""
    def __init__(self):
        self.latency_history = defaultdict(list)
        self.current_latencies = {}
        
    def add_measurement(self, source: str,
                       destination: str,
                       latency: float):
        """Add latency measurement"""
        key = f"{source}_{destination}"
        self.latency_history[key].append({
            'latency': latency,
            'timestamp': time.time()
        })
        
    def update(self):
        """Update latency measurements"""
        current_time = time.time()
        
        # Clean old history
        for key in self.latency_history:
            self.latency_history[key] = [
                m for m in self.latency_history[key]
                if current_time - m['timestamp'] < 300  # Keep last 5 minutes
            ]
            
            if self.latency_history[key]:
                # Update current latency (exponential moving average)
                alpha = 0.2
                current = self.latency_history[key][-1]['latency']
                previous = self.current_latencies.get(key, current)
                self.current_latencies[key] = alpha * current + (1 - alpha) * previous
                
    def get_current_latency(self, source: str,
                           destination: str) -> float:
        """Get current latency between nodes"""
        key = f"{source}_{destination}"
        return self.current_latencies.get(key, 0.0)
        
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics"""
        stats = {}
        for key, measurements in self.latency_history.items():
            if not measurements:
                continue
                
            latencies = [m['latency'] for m in measurements]
            stats[key] = {
                'current': self.current_latencies.get(key, 0.0),
                'average': np.mean(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'std': np.std(latencies)
            }
            
        return stats

class CompressionHandler:
    """Handle data compression"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compression_stats = defaultdict(list)
        
    def compress(self, data: Any) -> Any:
        """Compress data based on type"""
        if isinstance(data, torch.Tensor):
            return self._compress_tensor(data)
        elif isinstance(data, dict):
            return {k: self.compress(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.compress(v) for v in data)
        else:
            return data
            
    def decompress(self, data: Any) -> Any:
        """Decompress data based on type"""
        if isinstance(data, tuple) and len(data) == 2 and \
           isinstance(data[0], torch.Tensor):
            return self._decompress_tensor(data)
        elif isinstance(data, dict):
            return {k: self.decompress(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.decompress(v) for v in data)
        else:
            return data
            
    def _compress_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compress tensor using various techniques"""
        original_size = tensor.element_size() * tensor.nelement()
        
        # Quantization
        if self.config.get('use_quantization', True):
            tensor = self._quantize_tensor(tensor)
            
        # Sparsification
        if self.config.get('use_sparsification', True):
            tensor = self._sparsify_tensor(tensor)
            
        compressed_size = tensor.element_size() * tensor.nelement()
        
        # Track compression stats
        self.compression_stats['original'].append(original_size)
        self.compression_stats['compressed'].append(compressed_size)
        
        return (tensor, {
            'original_dtype': tensor.dtype,
            'original_shape': tensor.shape
        })
        
    def _decompress_tensor(self, data: Tuple[torch.Tensor, Dict]) -> torch.Tensor:
        """Decompress tensor"""
        tensor, metadata = data
        
        # Restore original properties
        tensor = tensor.to(dtype=metadata['original_dtype'])
        tensor = tensor.reshape(metadata['original_shape'])
        
        return tensor
        
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor values"""
        if tensor.dtype in [torch.float32, torch.float64]:
            return tensor.to(torch.float16)
        return tensor
        
    def _sparsify_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sparsify tensor by keeping top-k values"""
        if not self.config.get('sparsification_threshold', 0.0):
            return tensor
            
        threshold = self.config['sparsification_threshold']
        mask = torch.abs(tensor) > threshold
        return tensor * mask
        
    def get_stats(self) -> Dict[str, float]:
        """Get compression statistics"""
        if not self.compression_stats['original']:
            return {
                'ratio': 1.0,
                'savings': 0.0
            }
            
        total_original = sum(self.compression_stats['original'])
        total_compressed = sum(self.compression_stats['compressed'])
        
        ratio = total_compressed / total_original if total_original > 0 else 1.0
        savings = 1.0 - ratio
        
        return {
            'ratio': ratio,
            'savings': savings
        }
# task_generator.py
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Dict, Any
import math

class Task:
    """A single few-shot learning task with proper tensor types"""
    def __init__(self, support_x, support_y, query_x, query_y, task_info=None):
        # ALWAYS ensure proper types and no gradients for data
        self.support_x = support_x.float().detach().requires_grad_(False)
        self.support_y = support_y.long().detach().requires_grad_(False)
        self.query_x = query_x.float().detach().requires_grad_(False)
        self.query_y = query_y.long().detach().requires_grad_(False)
        
        # Ensure minimum batch dimensions
        if self.support_x.dim() == 1:
            self.support_x = self.support_x.unsqueeze(0)
        if self.query_x.dim() == 1:
            self.query_x = self.query_x.unsqueeze(0)
        if self.support_y.dim() == 0:
            self.support_y = self.support_y.unsqueeze(0)
        if self.query_y.dim() == 0:
            self.query_y = self.query_y.unsqueeze(0)
        
        self.task_info = task_info or {}
    
    def support(self):
        return self.support_x, self.support_y
    
    def query(self):
        return self.query_x, self.query_y

class TaskGenerator:
    """Generate diverse few-shot learning tasks with proper tensor handling"""
    
    def __init__(self, input_dim=768, num_classes=5, support_size=5, query_size=10):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.support_size = support_size
        self.query_size = query_size
        
        # Task type probabilities
        self.task_types = {
            'classification': 0.4,
            'pattern_recognition': 0.2,
            'sequence': 0.2,
            'logical': 0.1,
            'arithmetic': 0.1
        }
        
        # Domain templates for cross-domain transfer
        self.domains = {
            'source': {'feature_transform': self._identity_transform, 'noise_level': 0.1},
            'target': {'feature_transform': self._rotation_transform, 'noise_level': 0.15}
        }
        
    def sample(self):
        """Sample a random task"""
        task_type = np.random.choice(list(self.task_types.keys()), 
                                   p=list(self.task_types.values()))
        
        if task_type == 'classification':
            return self._create_classification_task()
        elif task_type == 'pattern_recognition':
            return self._create_pattern_task()
        elif task_type == 'sequence':
            return self._create_sequence_task()
        elif task_type == 'logical':
            return self._create_logical_task()
        else:  # arithmetic
            return self._create_arithmetic_task()
    
    def sample_from_domain(self, domain='source'):
        """Sample task from specific domain for transfer learning"""
        base_task = self.sample()
        domain_config = self.domains.get(domain, self.domains['source'])
        
        # Apply domain transformation
        transform = domain_config['feature_transform']
        noise_level = domain_config['noise_level']
        
        # Get base task data
        support_x, support_y = base_task.support()
        query_x, query_y = base_task.query()
        
        # Transform and add noise
        support_x_transformed = transform(support_x)
        support_x_transformed = support_x_transformed + torch.randn_like(support_x_transformed) * noise_level
        
        query_x_transformed = transform(query_x)
        query_x_transformed = query_x_transformed + torch.randn_like(query_x_transformed) * noise_level
        
        return Task(support_x_transformed, support_y, query_x_transformed, query_y,
                   task_info={'domain': domain, 'base_type': base_task.task_info.get('type', 'unknown')})
    
    def create_similar_task(self, original_task):
        """Create a task similar to the original (for interference testing)"""
        try:
            # Get original task data
            support_x, support_y = original_task.support()
            query_x, query_y = original_task.query()
            
            # Add small noise to inputs
            support_x_similar = support_x + torch.randn_like(support_x) * 0.1
            query_x_similar = query_x + torch.randn_like(query_x) * 0.1
            
            # Create label permutation
            unique_labels = torch.unique(support_y)
            num_unique = len(unique_labels)
            
            if num_unique > 1:
                # Create permutation
                perm = torch.randperm(num_unique)
                
                # Map labels
                label_map = torch.zeros(self.num_classes, dtype=torch.long)
                for i, old_label in enumerate(unique_labels):
                    new_label = unique_labels[perm[i]]
                    label_map[old_label] = new_label
                
                # Apply mapping
                support_y_similar = label_map[support_y.clamp(0, self.num_classes-1)]
                query_y_similar = label_map[query_y.clamp(0, self.num_classes-1)]
            else:
                # If only one label, use different random labels
                support_y_similar = torch.randint(0, self.num_classes, support_y.shape, dtype=torch.long)
                query_y_similar = torch.randint(0, self.num_classes, query_y.shape, dtype=torch.long)
            
            return Task(support_x_similar, support_y_similar, query_x_similar, query_y_similar,
                       task_info={'type': 'similar_interference'})
                       
        except Exception as e:
            # Fallback
            return self.sample()
    
    def create_emotional_task(self):
        """Create a task with high emotional salience"""
        base_task = self.sample()
        
        # Get base task data
        support_x, support_y = base_task.support()
        query_x, query_y = base_task.query()
        
        # Add emotional boost
        emotional_boost = torch.randn(self.input_dim) * 0.8
        
        support_x_emotional = support_x + emotional_boost.unsqueeze(0)
        query_x_emotional = query_x + emotional_boost.unsqueeze(0)
        
        return Task(support_x_emotional, support_y, query_x_emotional, query_y,
                   task_info={'type': 'emotional', 'salience': 'high'})
    
    def create_xor_task(self):
        """Create XOR-like logical task"""
        # Generate data
        support_x = torch.zeros(self.support_size, self.input_dim, dtype=torch.float32)
        query_x = torch.zeros(self.query_size, self.input_dim, dtype=torch.float32)
        
        # Binary features
        support_features = torch.randint(0, 2, (self.support_size, 2), dtype=torch.float32)
        query_features = torch.randint(0, 2, (self.query_size, 2), dtype=torch.float32)
        
        support_x[:, :2] = support_features
        query_x[:, :2] = query_features
        
        # Add correlated features
        for i in range(2, min(20, self.input_dim)):
            xor_signal_s = (support_features[:, 0] != support_features[:, 1]).float()
            support_x[:, i] = xor_signal_s * 2.0 + torch.randn(self.support_size) * 0.1
            
            xor_signal_q = (query_features[:, 0] != query_features[:, 1]).float()
            query_x[:, i] = xor_signal_q * 2.0 + torch.randn(self.query_size) * 0.1
        
        # Add noise
        if self.input_dim > 20:
            support_x[:, 20:] = torch.randn(self.support_size, self.input_dim - 20) * 0.1
            query_x[:, 20:] = torch.randn(self.query_size, self.input_dim - 20) * 0.1
        
        # XOR labels
        support_y = (support_features[:, 0] != support_features[:, 1]).long()
        query_y = (query_features[:, 0] != query_features[:, 1]).long()
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'xor_pattern'})
    
    def _create_sequence_task(self):
        """Create sequence completion task"""
        support_x = torch.zeros(self.support_size, self.input_dim, dtype=torch.float32)
        query_x = torch.zeros(self.query_size, self.input_dim, dtype=torch.float32)
        
        # Sequence parameters
        start = np.random.randint(1, 10)
        step = np.random.randint(1, 5)
        
        # Encode sequences
        for i in range(self.support_size):
            sequence_val = start + i * step
            support_x[i, 0] = sequence_val / 20.0
            support_x[i, 1] = i / 5.0
            support_x[i, 2] = (sequence_val ** 2) / 200.0
            support_x[i, 3] = np.sin(sequence_val / 5.0) * 2.0
            support_x[i, 4] = sequence_val % 3 / 3.0
            
        for i in range(self.query_size):
            sequence_val = start + (i + self.support_size) * step
            query_x[i, 0] = sequence_val / 20.0
            query_x[i, 1] = (i + self.support_size) / 5.0
            query_x[i, 2] = (sequence_val ** 2) / 200.0
            query_x[i, 3] = np.sin(sequence_val / 5.0) * 2.0
            query_x[i, 4] = sequence_val % 3 / 3.0
        
        # Add noise
        if self.input_dim > 5:
            support_x[:, 5:] = torch.randn(self.support_size, self.input_dim - 5) * 0.1
            query_x[:, 5:] = torch.randn(self.query_size, self.input_dim - 5) * 0.1
        
        # Labels are next values
        support_y = torch.tensor([(start + (i + 1) * step) % self.num_classes 
                                 for i in range(self.support_size)], dtype=torch.long)
        query_y = torch.tensor([(start + (i + self.support_size + 1) * step) % self.num_classes 
                               for i in range(self.query_size)], dtype=torch.long)
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'sequence', 'start': start, 'step': step})
    
    def _create_classification_task(self):
        """Create standard few-shot classification task"""
        # Generate centroids
        centroids = torch.randn(self.num_classes, self.input_dim, dtype=torch.float32) * 4.0
        
        # Ensure separation
        for i in range(1, self.num_classes):
            for j in range(i):
                while torch.norm(centroids[i] - centroids[j]) < 3.0:
                    centroids[i] = torch.randn(self.input_dim, dtype=torch.float32) * 4.0
        
        # Generate support set
        support_x = torch.zeros(self.support_size, self.input_dim, dtype=torch.float32)
        support_y = torch.zeros(self.support_size, dtype=torch.long)
        
        for i in range(self.support_size):
            class_idx = i % self.num_classes
            support_y[i] = class_idx
            support_x[i] = centroids[class_idx] + torch.randn(self.input_dim) * 0.2
        
        # Generate query set
        query_x = torch.zeros(self.query_size, self.input_dim, dtype=torch.float32)
        query_y = torch.zeros(self.query_size, dtype=torch.long)
        
        for i in range(self.query_size):
            class_idx = i % self.num_classes
            query_y[i] = class_idx
            query_x[i] = centroids[class_idx] + torch.randn(self.input_dim) * 0.2
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'classification', 'centroids': centroids})
    
    def _create_pattern_task(self):
        """Create pattern recognition task"""
        support_x = torch.randn(self.support_size, self.input_dim, dtype=torch.float32) * 0.3
        query_x = torch.randn(self.query_size, self.input_dim, dtype=torch.float32) * 0.3
        
        # Pattern based on distance from origin
        support_distances = torch.norm(support_x[:, :30], dim=1)
        query_distances = torch.norm(query_x[:, :30], dim=1)
        
        # Class boundaries
        distance_thresholds = torch.linspace(0.3, 4.0, self.num_classes + 1)
        
        support_y = torch.zeros(self.support_size, dtype=torch.long)
        query_y = torch.zeros(self.query_size, dtype=torch.long)
        
        # Assign classes
        for i, dist in enumerate(support_distances):
            for class_idx in range(self.num_classes):
                if distance_thresholds[class_idx] <= dist < distance_thresholds[class_idx + 1]:
                    support_y[i] = class_idx
                    break
            else:
                support_y[i] = self.num_classes - 1
                
        for i, dist in enumerate(query_distances):
            for class_idx in range(self.num_classes):
                if distance_thresholds[class_idx] <= dist < distance_thresholds[class_idx + 1]:
                    query_y[i] = class_idx
                    break
            else:
                query_y[i] = self.num_classes - 1
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'pattern_distance'})
    
    def _create_logical_task(self):
        """Create logical reasoning task"""
        support_x = torch.randn(self.support_size, self.input_dim, dtype=torch.float32) * 0.3
        query_x = torch.randn(self.query_size, self.input_dim, dtype=torch.float32) * 0.3
        
        # Add logical features
        for i in range(self.support_size):
            support_x[i, 0] = 2.0 if np.random.rand() > 0.5 else -2.0
            support_x[i, 1] = 2.0 if np.random.rand() > 0.5 else -2.0
            support_x[i, 2] = support_x[i, 0] * support_x[i, 1] / 4.0
            support_x[i, 3] = 2.0 if support_x[i, 0] > 0 or support_x[i, 1] > 0 else -2.0
            support_x[i, 4] = -support_x[i, 0]
            support_x[i, 5] = 2.0 if support_x[i, 0] > 0 and support_x[i, 1] < 0 else -2.0
            
        for i in range(self.query_size):
            query_x[i, 0] = 2.0 if np.random.rand() > 0.5 else -2.0
            query_x[i, 1] = 2.0 if np.random.rand() > 0.5 else -2.0
            query_x[i, 2] = query_x[i, 0] * query_x[i, 1] / 4.0
            query_x[i, 3] = 2.0 if query_x[i, 0] > 0 or query_x[i, 1] > 0 else -2.0
            query_x[i, 4] = -query_x[i, 0]
            query_x[i, 5] = 2.0 if query_x[i, 0] > 0 and query_x[i, 1] < 0 else -2.0
        
        # Labels based on logical rules
        support_condition = (support_x[:, 0] > 0) & (support_x[:, 1] > 0)
        query_condition = (query_x[:, 0] > 0) & (query_x[:, 1] > 0)
        
        support_y = support_condition.long()
        query_y = query_condition.long()
        
        # Extend to more classes
        if self.num_classes > 2:
            extra_condition_s = (support_x[:, 5] > 0)
            support_y = (support_y + extra_condition_s.long() * 2) % self.num_classes
            
            extra_condition_q = (query_x[:, 5] > 0)
            query_y = (query_y + extra_condition_q.long() * 2) % self.num_classes
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'logical_and'})
    
    def _create_arithmetic_task(self):
        """Create arithmetic reasoning task"""
        support_x = torch.zeros(self.support_size, self.input_dim, dtype=torch.float32)
        query_x = torch.zeros(self.query_size, self.input_dim, dtype=torch.float32)
        
        # Encode arithmetic problems
        for i in range(self.support_size):
            a = np.random.randint(1, 8)
            b = np.random.randint(1, 8)
            
            support_x[i, 0] = a / 5.0
            support_x[i, 1] = b / 5.0
            support_x[i, 2] = 1.0
            support_x[i, 3] = (a + b) / 10.0
            support_x[i, 4] = abs(a - b) / 8.0
            support_x[i, 5] = (a * b) / 50.0
            support_x[i, 6] = a % 3 / 3.0
            support_x[i, 7] = b % 3 / 3.0
            
            if self.input_dim > 8:
                support_x[i, 8:] = torch.randn(self.input_dim - 8) * 0.05
        
        for i in range(self.query_size):
            a = np.random.randint(1, 8)
            b = np.random.randint(1, 8)
            
            query_x[i, 0] = a / 5.0
            query_x[i, 1] = b / 5.0
            query_x[i, 2] = 1.0
            query_x[i, 3] = (a + b) / 10.0
            query_x[i, 4] = abs(a - b) / 8.0
            query_x[i, 5] = (a * b) / 50.0
            query_x[i, 6] = a % 3 / 3.0
            query_x[i, 7] = b % 3 / 3.0
            
            if self.input_dim > 8:
                query_x[i, 8:] = torch.randn(self.input_dim - 8) * 0.05
        
        # Labels are sums modulo num_classes
        support_y = torch.tensor([
            int((support_x[i, 0] * 5 + support_x[i, 1] * 5)) % self.num_classes
            for i in range(self.support_size)
        ], dtype=torch.long)
        
        query_y = torch.tensor([
            int((query_x[i, 0] * 5 + query_x[i, 1] * 5)) % self.num_classes
            for i in range(self.query_size)
        ], dtype=torch.long)
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'arithmetic_addition'})
    
    # Domain transformation functions
    def _identity_transform(self, x):
        return x.clone()
    
    def _rotation_transform(self, x):
        # Apply rotation to first few dimensions
        x_transformed = x.clone()
        if x.size(1) >= 4:
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = torch.tensor([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ], dtype=x.dtype)
            
            x_transformed[:, :2] = torch.mm(x[:, :2], rotation_matrix.T)
        return x_transformed

class MetaDataset:
    """Dataset wrapper for meta-learning tasks"""
    
    def __init__(self, task_generator, num_tasks=1000):
        self.task_generator = task_generator
        self.num_tasks = num_tasks
        self.tasks = []
        
        # Pre-generate tasks
        print(f"Generating {num_tasks} tasks...")
        for _ in range(num_tasks):
            self.tasks.append(task_generator.sample())
    
    def sample(self):
        """Sample a random task"""
        return random.choice(self.tasks)
    
    def get_task(self, idx):
        """Get specific task by index"""
        return self.tasks[idx % len(self.tasks)]
    
    def __len__(self):
        return len(self.tasks)

# Utility functions
def analyze_task_distribution(task_generator, num_samples=100):
    """Analyze the distribution of generated tasks"""
    task_types = {}
    
    for _ in range(num_samples):
        task = task_generator.sample()
        task_type = task.task_info.get('type', 'unknown')
        task_types[task_type] = task_types.get(task_type, 0) + 1
    
    print("Task Distribution:")
    for task_type, count in task_types.items():
        print(f"  {task_type}: {count} ({count/num_samples*100:.1f}%)")
    
    return task_types

def visualize_task_examples(task_generator, num_examples=3):
    """Visualize examples of different task types"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, num_examples, figsize=(15, 5))
        
        for i in range(num_examples):
            task = task_generator.sample()
            support_x, support_y = task.support()
            
            # Plot first two dimensions
            axes[i].scatter(support_x[:, 0].numpy(), support_x[:, 1].numpy(), 
                           c=support_y.numpy(), cmap='tab10')
            axes[i].set_title(f"Task Type: {task.task_info.get('type', 'unknown')}")
            axes[i].set_xlabel("Feature 1")
            axes[i].set_ylabel("Feature 2")
        
        plt.tight_layout()
        plt.savefig('task_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")

# Example usage
if __name__ == "__main__":
    # Create task generator
    task_gen = TaskGenerator(input_dim=768, num_classes=5, support_size=5, query_size=15)
    
    # Test basic functionality
    print("Testing Task Generator...")
    
    # Generate a few tasks
    for i in range(3):
        task = task_gen.sample()
        support_x, support_y = task.support()
        query_x, query_y = task.query()
        
        print(f"\nTask {i+1}:")
        print(f"  Type: {task.task_info.get('type', 'unknown')}")
        print(f"  Support set: {support_x.shape}, Labels: {support_y.shape}")
        print(f"  Query set: {query_x.shape}, Labels: {query_y.shape}")
        print(f"  Label distribution: {torch.bincount(support_y)}")
        print(f"  Tensor types: x={support_x.dtype}, y={support_y.dtype}")
    
    print("\nTask generator testing completed!")

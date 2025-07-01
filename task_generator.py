# task_generator.py
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Dict, Any
import math

class Task:
    """A single few-shot learning task"""
    def __init__(self, support_x, support_y, query_x, query_y, task_info=None):
        # FIXED: Ensure all tensors are proper types
        self.support_x = support_x.float() if support_x.dtype != torch.float32 else support_x
        self.support_y = support_y.long() if support_y.dtype != torch.long else support_y
        self.query_x = query_x.float() if query_x.dtype != torch.float32 else query_x
        self.query_y = query_y.long() if query_y.dtype != torch.long else query_y
        self.task_info = task_info or {}
    
    def support(self):
        return self.support_x, self.support_y
    
    def query(self):
        return self.query_x, self.query_y

class TaskGenerator:
    """Generate diverse few-shot learning tasks for meta-learning"""
    
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
        
        # Transform support set
        support_x = transform(base_task.support_x)
        support_x += torch.randn_like(support_x) * noise_level
        
        # Transform query set
        query_x = transform(base_task.query_x)
        query_x += torch.randn_like(query_x) * noise_level
        
        return Task(support_x, base_task.support_y, query_x, base_task.query_y,
                   task_info={'domain': domain, 'base_type': base_task.task_info.get('type', 'unknown')})
    
    def create_similar_task(self, original_task):
        """Create a task similar to the original (for interference testing)"""
        try:
            # Use similar input features but different labels
            support_x = original_task.support_x + torch.randn_like(original_task.support_x) * 0.1
            query_x = original_task.query_x + torch.randn_like(original_task.query_x) * 0.1
            
            # Create safe label permutation
            unique_labels = torch.unique(original_task.support_y)
            num_unique = len(unique_labels)
            
            if num_unique > 1:
                # Create permutation for the unique labels
                label_permutation = torch.randperm(num_unique)
                
                # Create mapping from old labels to new labels
                label_map = torch.zeros(self.num_classes, dtype=torch.long)
                for i, old_label in enumerate(unique_labels):
                    new_label = unique_labels[label_permutation[i]]
                    label_map[old_label] = new_label
                
                # Apply mapping safely
                support_y = torch.zeros_like(original_task.support_y)
                query_y = torch.zeros_like(original_task.query_y)
                
                for i in range(len(support_y)):
                    old_label = original_task.support_y[i].item()
                    if old_label < len(label_map):
                        support_y[i] = label_map[old_label]
                    else:
                        support_y[i] = old_label
                
                for i in range(len(query_y)):
                    old_label = original_task.query_y[i].item()
                    if old_label < len(label_map):
                        query_y[i] = label_map[old_label]
                    else:
                        query_y[i] = old_label
            else:
                # If only one unique label, just use different random labels
                support_y = torch.randint(0, self.num_classes, original_task.support_y.shape)
                query_y = torch.randint(0, self.num_classes, original_task.query_y.shape)
            
            return Task(support_x, support_y, query_x, query_y,
                       task_info={'type': 'similar_interference'})
                       
        except Exception as e:
            # Fallback: create a completely new task if similarity creation fails
            return self.sample()
    
    def create_emotional_task(self):
        """Create a task with high emotional salience"""
        base_task = self.sample()
        
        # Add emotional context by amplifying certain features
        emotional_boost = torch.randn(self.input_dim) * 0.5  # INCREASED for more salience
        
        support_x = base_task.support_x + emotional_boost.unsqueeze(0)
        query_x = base_task.query_x + emotional_boost.unsqueeze(0)
        
        return Task(support_x, base_task.support_y, query_x, base_task.query_y,
                   task_info={'type': 'emotional', 'salience': 'high'})
    
    def create_xor_task(self):
        """Create XOR-like logical task"""
        # Generate binary feature pairs
        support_x = torch.zeros(self.support_size, self.input_dim)
        query_x = torch.zeros(self.query_size, self.input_dim)
        
        # Set first two dimensions as binary features
        support_features = torch.randint(0, 2, (self.support_size, 2)).float()
        query_features = torch.randint(0, 2, (self.query_size, 2)).float()
        
        support_x[:, :2] = support_features
        query_x[:, :2] = query_features
        
        # Add meaningful signal to more dimensions for better learning
        for i in range(2, min(10, self.input_dim)):
            support_x[:, i] = support_features[:, 0] * support_features[:, 1] + torch.randn(self.support_size) * 0.1
            query_x[:, i] = query_features[:, 0] * query_features[:, 1] + torch.randn(self.query_size) * 0.1
        
        # Add noise to remaining dimensions
        if self.input_dim > 10:
            support_x[:, 10:] = torch.randn(self.support_size, self.input_dim - 10) * 0.1
            query_x[:, 10:] = torch.randn(self.query_size, self.input_dim - 10) * 0.1
        
        # XOR labels
        support_y = (support_features[:, 0] != support_features[:, 1]).long()
        query_y = (query_features[:, 0] != query_features[:, 1]).long()
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'xor_pattern'})
    
    def _create_sequence_task(self):
        """Create sequence completion task"""
        # Generate arithmetic sequences
        support_x = torch.zeros(self.support_size, self.input_dim)
        query_x = torch.zeros(self.query_size, self.input_dim)
        
        # Random arithmetic sequence parameters
        start = np.random.randint(1, 10)
        step = np.random.randint(1, 5)
        
        # Encode sequence positions in first few dimensions with more signal
        for i in range(self.support_size):
            sequence_val = start + i * step
            support_x[i, 0] = sequence_val / 50.0  # Normalize
            support_x[i, 1] = i / 10.0  # Position encoding
            # Add more sequence-related features
            support_x[i, 2] = (sequence_val ** 2) / 500.0  # Quadratic feature
            support_x[i, 3] = np.sin(sequence_val / 10.0)  # Periodic feature
            
        for i in range(self.query_size):
            sequence_val = start + (i + self.support_size) * step
            query_x[i, 0] = sequence_val / 50.0
            query_x[i, 1] = (i + self.support_size) / 10.0
            query_x[i, 2] = (sequence_val ** 2) / 500.0
            query_x[i, 3] = np.sin(sequence_val / 10.0)
        
        # Add noise to remaining dimensions
        if self.input_dim > 4:
            support_x[:, 4:] = torch.randn(self.support_size, self.input_dim - 4) * 0.1
            query_x[:, 4:] = torch.randn(self.query_size, self.input_dim - 4) * 0.1
        
        # Labels are next values in sequence (mod num_classes)
        support_y = torch.tensor([(start + (i + 1) * step) % self.num_classes 
                                 for i in range(self.support_size)], dtype=torch.long)
        query_y = torch.tensor([(start + (i + self.support_size + 1) * step) % self.num_classes 
                               for i in range(self.query_size)], dtype=torch.long)
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'sequence', 'start': start, 'step': step})
    
    def _create_classification_task(self):
        """Create standard few-shot classification task"""
        # Generate random centroids for each class with better separation
        centroids = torch.randn(self.num_classes, self.input_dim) * 3.0  # INCREASED separation
        
        # Ensure minimum separation between centroids
        for i in range(1, self.num_classes):
            for j in range(i):
                while torch.norm(centroids[i] - centroids[j]) < 2.0:  # Minimum distance
                    centroids[i] = torch.randn(self.input_dim) * 3.0
        
        # Generate support set with balanced classes
        support_x = torch.zeros(self.support_size, self.input_dim)
        support_y = torch.zeros(self.support_size, dtype=torch.long)
        
        samples_per_class = max(1, self.support_size // self.num_classes)
        
        for i in range(self.support_size):
            class_idx = i % self.num_classes
            support_y[i] = class_idx
            
            # Sample around centroid with controlled noise
            support_x[i] = centroids[class_idx] + torch.randn(self.input_dim) * 0.3  # REDUCED noise
        
        # Generate query set with balanced classes
        query_x = torch.zeros(self.query_size, self.input_dim)
        query_y = torch.zeros(self.query_size, dtype=torch.long)
        
        for i in range(self.query_size):
            class_idx = i % self.num_classes  # CHANGED: Balanced instead of random
            query_y[i] = class_idx
            query_x[i] = centroids[class_idx] + torch.randn(self.input_dim) * 0.3
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'classification', 'centroids': centroids})
    
    def _create_pattern_task(self):
        """Create pattern recognition task"""
        # Create tasks based on geometric patterns in feature space
        support_x = torch.randn(self.support_size, self.input_dim) * 0.5
        query_x = torch.randn(self.query_size, self.input_dim) * 0.5
        
        # Pattern: distance from origin determines class - IMPROVED
        support_distances = torch.norm(support_x[:, :20], dim=1)  # Use more dimensions
        query_distances = torch.norm(query_x[:, :20], dim=1)
        
        # Create more meaningful class boundaries
        distance_thresholds = torch.linspace(0.5, 3.0, self.num_classes + 1)
        
        support_y = torch.zeros(self.support_size, dtype=torch.long)
        query_y = torch.zeros(self.query_size, dtype=torch.long)
        
        for i, dist in enumerate(support_distances):
            for class_idx in range(self.num_classes):
                if distance_thresholds[class_idx] <= dist < distance_thresholds[class_idx + 1]:
                    support_y[i] = class_idx
                    break
            else:
                support_y[i] = self.num_classes - 1  # Last class for very large distances
                
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
        # Simple logical rules based on feature combinations
        support_x = torch.randn(self.support_size, self.input_dim) * 0.5
        query_x = torch.randn(self.query_size, self.input_dim) * 0.5
        
        # Add clear logical structure to first few dimensions
        for i in range(self.support_size):
            # Create logical pattern in first 4 dimensions
            support_x[i, 0] = 1.0 if np.random.rand() > 0.5 else -1.0  # Binary feature A
            support_x[i, 1] = 1.0 if np.random.rand() > 0.5 else -1.0  # Binary feature B
            support_x[i, 2] = support_x[i, 0] * support_x[i, 1]  # A AND B
            support_x[i, 3] = 1.0 if support_x[i, 0] > 0 or support_x[i, 1] > 0 else -1.0  # A OR B
            
        for i in range(self.query_size):
            query_x[i, 0] = 1.0 if np.random.rand() > 0.5 else -1.0
            query_x[i, 1] = 1.0 if np.random.rand() > 0.5 else -1.0
            query_x[i, 2] = query_x[i, 0] * query_x[i, 1]
            query_x[i, 3] = 1.0 if query_x[i, 0] > 0 or query_x[i, 1] > 0 else -1.0
        
        # Rule: if feature[0] > 0 AND feature[1] > 0, then class 1, else class 0
        support_condition = (support_x[:, 0] > 0) & (support_x[:, 1] > 0)
        query_condition = (query_x[:, 0] > 0) & (query_x[:, 1] > 0)
        
        support_y = support_condition.long()
        query_y = query_condition.long()
        
        # Extend to more classes if needed
        if self.num_classes > 2:
            # Add more complex rules
            extra_condition = (support_x[:, 2] > 0.5)
            support_y = (support_y + extra_condition.long() * 2) % self.num_classes
            
            extra_condition_q = (query_x[:, 2] > 0.5)
            query_y = (query_y + extra_condition_q.long() * 2) % self.num_classes
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'logical_and'})
    
    def _create_arithmetic_task(self):
        """Create arithmetic reasoning task"""
        support_x = torch.zeros(self.support_size, self.input_dim)
        query_x = torch.zeros(self.query_size, self.input_dim)
        
        # Encode simple arithmetic problems in features
        for i in range(self.support_size):
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 10)
            
            support_x[i, 0] = a / 10.0  # First operand
            support_x[i, 1] = b / 10.0  # Second operand
            support_x[i, 2] = 1.0  # Operation encoding (addition)
            support_x[i, 3] = (a + b) / 20.0  # Expected result (normalized)
            support_x[i, 4] = abs(a - b) / 10.0  # Difference
            support_x[i, 5] = (a * b) / 100.0  # Product
            
            # Add noise to remaining dimensions
            if self.input_dim > 6:
                support_x[i, 6:] = torch.randn(self.input_dim - 6) * 0.1
        
        for i in range(self.query_size):
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 10)
            
            query_x[i, 0] = a / 10.0
            query_x[i, 1] = b / 10.0
            query_x[i, 2] = 1.0
            query_x[i, 3] = (a + b) / 20.0
            query_x[i, 4] = abs(a - b) / 10.0
            query_x[i, 5] = (a * b) / 100.0
            
            if self.input_dim > 6:
                query_x[i, 6:] = torch.randn(self.input_dim - 6) * 0.1
        
        # Labels are sums modulo num_classes
        support_y = torch.tensor([
            int((support_x[i, 0] * 10 + support_x[i, 1] * 10)) % self.num_classes
            for i in range(self.support_size)
        ], dtype=torch.long)
        
        query_y = torch.tensor([
            int((query_x[i, 0] * 10 + query_x[i, 1] * 10)) % self.num_classes
            for i in range(self.query_size)
        ], dtype=torch.long)
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'arithmetic_addition'})
    
    # Domain transformation functions
    def _identity_transform(self, x):
        return x
    
    def _rotation_transform(self, x):
        # Apply random rotation to first few dimensions
        if x.size(1) >= 4:
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = torch.tensor([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ], dtype=x.dtype)
            
            x_rotated = x.clone()
            x_rotated[:, :2] = torch.mm(x[:, :2], rotation_matrix.T)
            return x_rotated
        return x

class MetaDataset:
    """Dataset wrapper for meta-learning tasks"""
    
    def __init__(self, task_generator, num_tasks=1000):
        self.task_generator = task_generator
        self.num_tasks = num_tasks
        self.tasks = []
        
        # Pre-generate tasks for consistency
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

# Utility functions for task analysis
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
            
            # Plot first two dimensions of support set
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

# Example usage and testing
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
    
    # Test different domains
    print("\nTesting domain transfer...")
    source_task = task_gen.sample_from_domain('source')
    target_task = task_gen.sample_from_domain('target')
    
    print(f"Source domain task: {source_task.task_info}")
    print(f"Target domain task: {target_task.task_info}")
    
    # Test similar task generation
    print("\nTesting similar task generation...")
    original = task_gen.sample()
    similar = task_gen.create_similar_task(original)
    
    print(f"Original task type: {original.task_info.get('type', 'unknown')}")
    print(f"Similar task type: {similar.task_info.get('type', 'unknown')}")
    
    # Analyze task distribution
    print("\nAnalyzing task distribution...")
    analyze_task_distribution(task_gen, num_samples=50)
    
    print("\nTask generator testing completed!")

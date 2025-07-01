# meta_training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import logging
from tqdm import tqdm
import copy

def enhanced_meta_train(model, tasks, inner_steps=5, inner_lr=1e-2, outer_lr=1e-3, 
                       iters=1000, forgetting_freq=10, consolidation_freq=100,
                       evaluation_freq=50, log_freq=25):
    """
    Enhanced meta-training with human-like forgetting and consolidation
    FIXED to ensure proper memory usage and learning
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Optimizers
    opt = torch.optim.Adam(model.parameters(), lr=outer_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    
    # Training statistics
    stats = {
        'meta_losses': [],
        'inner_losses': [],
        'memory_stats': [],
        'forgetting_events': [],
        'consolidation_events': [],
        'test_accuracies': []
    }
    
    # Memory for tracking forgetting curves
    memory_tracker = MemoryTracker()
    
    logger.info(f"Starting enhanced meta-training for {iters} iterations")
    
    for i in tqdm(range(iters), desc="Meta-training"):
        # Sample a task
        task = tasks.sample()
        model.train()
        
        # ========== IMPROVED MAML WITH MEMORY FOCUS ==========
        
        # Get task data
        x_supp, y_supp = task.support()
        x_qry, y_qry = task.query()
        
        # Ensure proper tensor types
        y_supp = y_supp.long()
        y_qry = y_qry.long()
        
        # ========== INNER LOOP: Task-specific adaptation ==========
        
        # Store original parameters for gradient computation
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        inner_losses = []
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
        
        for inner_step in range(inner_steps):
            # Forward pass with FORCED memory updates
            y_pred, aux_info = model(x_supp, update_memory=True)
            
            # Task-specific loss
            inner_loss = F.cross_entropy(y_pred, y_supp)
            inner_losses.append(inner_loss.item())
            
            # Inner gradient step
            inner_optimizer.zero_grad()
            inner_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            inner_optimizer.step()
            
            # FORCE memory consolidation during inner loop occasionally
            if inner_step == inner_steps - 1:  # Last inner step
                model.mem.consolidate_memories()
        
        # ========== OUTER LOOP: Meta-update ==========
        
        # Forward pass for meta-loss (keep memory updates for learning)
        y_pred_qry, aux_info = model(x_qry, update_memory=True)
        
        # Meta-loss
        meta_loss = F.cross_entropy(y_pred_qry, y_qry)
        
        # Add memory regularization to encourage usage
        memory_reg = compute_memory_regularization(model)
        memory_usage_bonus = compute_memory_usage_bonus(model)  # NEW: Reward memory usage
        
        total_loss = meta_loss + 0.001 * memory_reg - 0.01 * memory_usage_bonus
        
        # Meta-update
        opt.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()
        scheduler.step()
        
        # ========== MEMORY MANAGEMENT ==========
        
        # Apply forgetting mechanisms - LESS FREQUENT
        if i % forgetting_freq == 0 and i > 50:  # Start forgetting after some learning
            pre_forget_stats = model.get_memory_stats()
            model.forget_and_consolidate()
            post_forget_stats = model.get_memory_stats()
            
            # Track forgetting event
            forgetting_event = {
                'iteration': i,
                'schemas_before': pre_forget_stats['active_schemas'],
                'schemas_after': post_forget_stats['active_schemas'],
                'schemas_forgotten': max(0, pre_forget_stats['active_schemas'] - post_forget_stats['active_schemas'])
            }
            stats['forgetting_events'].append(forgetting_event)
            
            if i % log_freq == 0:
                logger.info(f"Iter {i}: Forgot {forgetting_event['schemas_forgotten']} schemas, "
                          f"{post_forget_stats['active_schemas']} remain active")
        
        # Memory consolidation - MORE FREQUENT
        if i % consolidation_freq == 0 and i > 0:
            pre_consolidation_stats = model.get_memory_stats()
            model.mem.consolidate_memories()
            post_consolidation_stats = model.get_memory_stats()
            
            consolidation_event = {
                'iteration': i,
                'avg_importance_before': pre_consolidation_stats.get('avg_schema_importance', 0),
                'avg_importance_after': post_consolidation_stats.get('avg_schema_importance', 0)
            }
            stats['consolidation_events'].append(consolidation_event)
            
            logger.info(f"Iter {i}: Memory consolidation completed")
        
        # ========== LOGGING AND EVALUATION ==========
        
        # Record statistics
        stats['meta_losses'].append(meta_loss.item())
        stats['inner_losses'].append(np.mean(inner_losses))
        
        if i % log_freq == 0:
            current_stats = model.get_memory_stats()
            stats['memory_stats'].append({
                'iteration': i,
                **current_stats
            })
            
            # Track memory evolution
            memory_tracker.update(i, current_stats)
        
        # Evaluation on test tasks
        if i % evaluation_freq == 0 and i > 0:
            test_acc = evaluate_model(model, tasks, num_test_tasks=5)  # Reduced for quick demo
            stats['test_accuracies'].append({
                'iteration': i,
                'accuracy': test_acc
            })
            
            logger.info(f"Iter {i}: Test accuracy = {test_acc:.3f}")
        
        # Detailed logging
        if i % log_freq == 0:
            current_lr = scheduler.get_last_lr()[0]
            current_stats = model.get_memory_stats()
            logger.info(f"Iter {i}: Meta loss = {meta_loss.item():.4f}, "
                       f"Inner loss = {np.mean(inner_losses):.4f}, "
                       f"Active schemas = {current_stats['active_schemas']}, "
                       f"WM slots = {current_stats['working_memory_slots_used']}, "
                       f"LR = {current_lr:.6f}")
    
    logger.info("Meta-training completed!")
    
    # Generate training report
    generate_training_report(model, stats, memory_tracker)
    
    return model, stats

def compute_memory_regularization(model):
    """Compute regularization term based on memory usage"""
    try:
        # Encourage efficient memory usage but not too much
        wm_usage = model.wm.slot_occupied.float().mean()
        schema_usage = model.mem.schema_active.float().mean()
        
        # Light penalty for extreme overuse, but encourage some usage
        memory_penalty = 0.1 * torch.relu(wm_usage - 0.8) + 0.1 * torch.relu(schema_usage - 0.8)
        
        return memory_penalty
    except:
        # Fallback if memory stats are not available
        return torch.tensor(0.0)

def compute_memory_usage_bonus(model):
    """NEW: Reward memory usage to encourage learning"""
    try:
        # Reward active memory usage
        wm_usage = model.wm.slot_occupied.float().mean()
        schema_usage = model.mem.schema_active.float().mean()
        
        # Bonus for having active memories (up to a reasonable limit)
        usage_bonus = torch.relu(wm_usage) + torch.relu(schema_usage)
        
        return usage_bonus
    except:
        return torch.tensor(0.0)

def evaluate_model(model, tasks, num_test_tasks=10):
    """Evaluate model on test tasks with proper gradient handling"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for _ in range(num_test_tasks):
            try:
                task = tasks.sample()
                
                # Get task data
                x_supp, y_supp = task.support()
                x_qry, y_qry = task.query()
                
                # Ensure proper tensor types
                y_supp = y_supp.long()
                y_qry = y_qry.long()
                
                # Quick adaptation on support set WITH memory updates
                temp_model = copy.deepcopy(model)
                temp_optimizer = torch.optim.SGD(temp_model.parameters(), lr=0.01)
                
                for _ in range(3):
                    temp_model.train()
                    y_pred, _ = temp_model(x_supp, update_memory=True)
                    loss = F.cross_entropy(y_pred, y_supp)
                    temp_optimizer.zero_grad()
                    loss.backward()
                    temp_optimizer.step()
                
                # Evaluate on query set
                temp_model.eval()
                y_pred_qry, _ = temp_model(x_qry, update_memory=False)
                
                predicted = torch.argmax(y_pred_qry, dim=1)
                total_correct += (predicted == y_qry).sum().item()
                total_samples += y_qry.size(0)
                
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Evaluation task failed: {e}")
                continue
    
    model.train()
    return total_correct / total_samples if total_samples > 0 else 0.0

class MemoryTracker:
    """Track memory evolution over training"""
    def __init__(self):
        self.memory_evolution = defaultdict(list)
        self.forgetting_curves = defaultdict(list)
    
    def update(self, iteration, memory_stats):
        """Update memory tracking statistics"""
        for key, value in memory_stats.items():
            if value is not None and not np.isnan(float(value)):
                self.memory_evolution[key].append((iteration, value))
    
    def plot_memory_evolution(self):
        """Plot memory statistics over training"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Active schemas over time
            if 'active_schemas' in self.memory_evolution and self.memory_evolution['active_schemas']:
                iterations, active_schemas = zip(*self.memory_evolution['active_schemas'])
                axes[0, 0].plot(iterations, active_schemas)
                axes[0, 0].set_title('Active Schemas Over Time')
                axes[0, 0].set_xlabel('Iteration')
                axes[0, 0].set_ylabel('Number of Active Schemas')
                axes[0, 0].grid(True)
            
            # Average importance over time
            if 'avg_schema_importance' in self.memory_evolution and self.memory_evolution['avg_schema_importance']:
                iterations, importance = zip(*self.memory_evolution['avg_schema_importance'])
                axes[0, 1].plot(iterations, importance)
                axes[0, 1].set_title('Average Schema Importance')
                axes[0, 1].set_xlabel('Iteration')
                axes[0, 1].set_ylabel('Importance Score')
                axes[0, 1].grid(True)
            
            # Working memory usage
            if 'working_memory_slots_used' in self.memory_evolution and self.memory_evolution['working_memory_slots_used']:
                iterations, wm_usage = zip(*self.memory_evolution['working_memory_slots_used'])
                axes[1, 0].plot(iterations, wm_usage)
                axes[1, 0].set_title('Working Memory Usage')
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('Slots Used')
                axes[1, 0].grid(True)
            
            # Memory access patterns
            if 'avg_access_count' in self.memory_evolution and self.memory_evolution['avg_access_count']:
                iterations, access_count = zip(*self.memory_evolution['avg_access_count'])
                axes[1, 1].plot(iterations, access_count)
                axes[1, 1].set_title('Average Memory Access Count')
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Access Count')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig('memory_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close to prevent display issues
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to plot memory evolution: {e}")

def test_forgetting_curves(model, initial_tasks, retention_intervals=[1, 5, 10, 25, 50]):
    """Test human-like forgetting curves"""
    model.eval()
    forgetting_data = []
    
    with torch.no_grad():
        # Learn initial tasks
        for task in initial_tasks:
            x_supp, y_supp = task.support()
            y_supp = y_supp.long()
            
            # Store initial performance
            y_pred, _ = model(x_supp, update_memory=False)
            initial_acc = (torch.argmax(y_pred, dim=1) == y_supp).float().mean().item()
            
            # Test retention at different intervals
            for interval in retention_intervals:
                # Create model copy for testing
                test_model = copy.deepcopy(model)
                
                # Simulate time passing
                for _ in range(interval):
                    test_model.forget_and_consolidate()
                
                # Test retention
                y_pred, _ = test_model(x_supp, update_memory=False)
                retention_acc = (torch.argmax(y_pred, dim=1) == y_supp).float().mean().item()
                
                forgetting_data.append({
                    'interval': interval,
                    'initial_accuracy': initial_acc,
                    'retention_accuracy': retention_acc,
                    'retention_ratio': retention_acc / (initial_acc + 1e-8)
                })
    
    return forgetting_data

def generate_training_report(model, stats, memory_tracker):
    """Generate comprehensive training report"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*50)
    logger.info("ENHANCED UNDERSTANDING MODEL TRAINING REPORT")
    logger.info("="*50)
    
    try:
        # Final model statistics
        final_stats = model.get_memory_stats()
        logger.info(f"\nFinal Model Statistics:")
        logger.info(f"  Active Schemas: {final_stats.get('active_schemas', 0)}")
        logger.info(f"  Working Memory Usage: {final_stats.get('working_memory_slots_used', 0)}/7 slots")
        logger.info(f"  Average Schema Importance: {final_stats.get('avg_schema_importance', 0):.3f}")
        logger.info(f"  Training Steps: {final_stats.get('total_training_steps', 0)}")
        
        # Training dynamics
        if stats['test_accuracies']:
            final_acc = stats['test_accuracies'][-1]['accuracy']
            logger.info(f"\nFinal Test Accuracy: {final_acc:.3f}")
        
        # Memory dynamics
        total_forgotten = sum(event['schemas_forgotten'] for event in stats['forgetting_events'])
        logger.info(f"\nMemory Dynamics:")
        logger.info(f"  Total Forgetting Events: {len(stats['forgetting_events'])}")
        logger.info(f"  Total Schemas Forgotten: {total_forgotten}")
        logger.info(f"  Consolidation Events: {len(stats['consolidation_events'])}")
        
        # Check if memory system is working
        max_schemas = max([stat.get('active_schemas', 0) for stat in stats['memory_stats']], default=0)
        max_wm = max([stat.get('working_memory_slots_used', 0) for stat in stats['memory_stats']], default=0)
        
        logger.info(f"\nMemory System Health:")
        logger.info(f"  Max Schemas Used: {max_schemas}")
        logger.info(f"  Max WM Slots Used: {max_wm}")
        logger.info(f"  Memory System Active: {'✓' if max_schemas > 0 or max_wm > 0 else '✗'}")
        
        # Plot memory evolution
        memory_tracker.plot_memory_evolution()
        
        logger.info("\nTraining completed successfully!")
        
    except Exception as e:
        logger.warning(f"Error in training report: {e}")
    
    logger.info("="*50)

# Example usage and testing
if __name__ == "__main__":
    from understanding_model import EnhancedUnderstandingNet
    
    # Create model
    model = EnhancedUnderstandingNet(d_model=256, num_classes=5)
    
    # Dummy task sampler (replace with real task distribution)
    class DummyTaskSampler:
        def sample(self):
            # Return dummy task with support and query sets
            class DummyTask:
                def support(self):
                    return torch.randn(5, 768), torch.randint(0, 5, (5,))
                def query(self):
                    return torch.randn(10, 768), torch.randint(0, 5, (10,))
            return DummyTask()
    
    tasks = DummyTaskSampler()
    
    # Train model
    trained_model, training_stats = enhanced_meta_train(
        model=model,
        tasks=tasks,
        iters=200,
        inner_steps=3,
        inner_lr=1e-2,
        outer_lr=1e-3
    )
    
    print("Training completed successfully!")

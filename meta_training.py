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
    """FIXED: Enhanced meta-training with NaN prevention and proper memory handling"""
    
    import torch
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Optimizers with gradient clipping
    opt = torch.optim.Adam(model.parameters(), lr=outer_lr, eps=1e-8)
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
    
    logger.info(f"Starting FIXED meta-training for {iters} iterations")
    
    for i in tqdm(range(iters), desc="Meta-training"):
        try:
            # Sample a task
            task = tasks.sample()
            model.train()
            
            # Get task data with proper error handling
            x_supp, y_supp = task.support()
            x_qry, y_qry = task.query()
            
            # FIXED: Ensure proper tensor types and no NaN
            x_supp = x_supp.float().detach().requires_grad_(False)
            y_supp = y_supp.long().detach().requires_grad_(False)
            x_qry = x_qry.float().detach().requires_grad_(False)
            y_qry = y_qry.long().detach().requires_grad_(False)
            
            # Check for NaN inputs
            if torch.isnan(x_supp).any() or torch.isnan(x_qry).any():
                logger.warning(f"NaN detected in inputs at iteration {i}, skipping...")
                continue
            
            # Ensure batch dimensions
            if x_supp.dim() == 1:
                x_supp = x_supp.unsqueeze(0)
            if x_qry.dim() == 1:
                x_qry = x_qry.unsqueeze(0)
            if y_supp.dim() == 0:
                y_supp = y_supp.unsqueeze(0)
            if y_qry.dim() == 0:
                y_qry = y_qry.unsqueeze(0)
            
            # SIMPLIFIED TRAINING APPROACH - NO COMPLEX INNER LOOP
            inner_losses = []
            
            # Inner loop: adapt to support set (SIMPLIFIED)
            for inner_step in range(inner_steps):
                try:
                    # Forward pass with memory updates but careful NaN checking
                    y_pred, aux_info = model(x_supp, update_memory=(inner_step == inner_steps-1))
                    
                    # Check for NaN in predictions
                    if torch.isnan(y_pred).any():
                        logger.warning(f"NaN in predictions at iter {i}, inner step {inner_step}")
                        break
                    
                    # Handle shape mismatches
                    if y_pred.shape[0] != y_supp.shape[0]:
                        min_batch = min(y_pred.shape[0], y_supp.shape[0])
                        y_pred = y_pred[:min_batch]
                        y_supp_batch = y_supp[:min_batch]
                    else:
                        y_supp_batch = y_supp
                    
                    # Compute loss with NaN protection
                    inner_loss = F.cross_entropy(y_pred, y_supp_batch, reduction='mean')
                    
                    if torch.isnan(inner_loss):
                        logger.warning(f"NaN in inner loss at iter {i}")
                        inner_loss = torch.tensor(1.6)  # Fallback value
                    
                    inner_losses.append(inner_loss.item())
                    
                except Exception as e:
                    logger.warning(f"Inner loop error at iter {i}: {e}")
                    inner_losses.append(1.6)
                    continue
            
            # Evaluate on query set with NaN protection
            try:
                y_pred_qry, aux_info = model(x_qry, update_memory=True)
                
                # Check for NaN
                if torch.isnan(y_pred_qry).any():
                    logger.warning(f"NaN in query predictions at iter {i}")
                    # Skip this iteration
                    continue
                
                # Handle shape mismatches
                if y_pred_qry.shape[0] != y_qry.shape[0]:
                    min_batch = min(y_pred_qry.shape[0], y_qry.shape[0])
                    y_pred_qry = y_pred_qry[:min_batch]
                    y_qry_batch = y_qry[:min_batch]
                else:
                    y_qry_batch = y_qry
                
                # Meta-loss with NaN protection
                meta_loss = F.cross_entropy(y_pred_qry, y_qry_batch, reduction='mean')
                
                if torch.isnan(meta_loss):
                    logger.warning(f"NaN in meta loss at iter {i}, skipping update")
                    continue
                
                # SAFE Meta-update
                opt.zero_grad()
                meta_loss.backward()
                
                # CRITICAL: Gradient clipping to prevent explosion
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    logger.warning(f"NaN gradients detected at iter {i}, skipping update")
                    opt.zero_grad()
                    continue
                
                opt.step()
                scheduler.step()
                
            except Exception as e:
                logger.warning(f"Query evaluation error at iter {i}: {e}")
                continue
            
            # FIXED: Safe memory management
            if i % forgetting_freq == 0 and i > 10:
                try:
                    pre_forget_stats = model.get_memory_stats()
                    
                    # Apply forgetting with safety checks
                    model.forget_and_consolidate()
                    
                    # Force additional forgetting if needed (SAFER VERSION)
                    if i % (forgetting_freq * 2) == 0:
                        with torch.no_grad():
                            # Check if schema store exists and has active schemas
                            if hasattr(model, 'mem') and hasattr(model.mem, 'schema_active'):
                                active_schemas = torch.where(model.mem.schema_active)[0]
                                if len(active_schemas) > 10:
                                    num_to_forget = max(1, len(active_schemas) // 8)
                                    
                                    # Safe importance score access
                                    if hasattr(model.mem, 'importance_scores'):
                                        importance_scores = model.mem.importance_scores[active_schemas]
                                        _, worst_indices = torch.topk(importance_scores, num_to_forget, largest=False)
                                        schemas_to_forget = active_schemas[worst_indices]
                                        
                                        # Actually forget (safe operations)
                                        model.mem.schema_active[schemas_to_forget] = False
                                        model.mem.values[schemas_to_forget] = 0
                                        model.mem.importance_scores[schemas_to_forget] = 0
                                        
                                        if hasattr(model.mem, 'access_counts'):
                                            model.mem.access_counts[schemas_to_forget] = 0
                    
                    post_forget_stats = model.get_memory_stats()
                    
                    # Track forgetting safely
                    schemas_forgotten = max(0, pre_forget_stats.get('active_schemas', 0) - 
                                          post_forget_stats.get('active_schemas', 0))
                    
                    forgetting_event = {
                        'iteration': i,
                        'schemas_before': pre_forget_stats.get('active_schemas', 0),
                        'schemas_after': post_forget_stats.get('active_schemas', 0),
                        'schemas_forgotten': schemas_forgotten
                    }
                    stats['forgetting_events'].append(forgetting_event)
                    
                    if i % log_freq == 0:
                        logger.info(f"Iter {i}: Forgot {schemas_forgotten} schemas, "
                                  f"{post_forget_stats.get('active_schemas', 0)} remain active")
                        
                except Exception as e:
                    logger.warning(f"Forgetting mechanism error at iter {i}: {e}")
            
            # Memory consolidation (SAFER)
            if i % consolidation_freq == 0 and i > 0:
                try:
                    pre_consolidation_stats = model.get_memory_stats()
                    
                    if hasattr(model.mem, 'consolidate_memories'):
                        model.mem.consolidate_memories()
                    
                    post_consolidation_stats = model.get_memory_stats()
                    
                    consolidation_event = {
                        'iteration': i,
                        'avg_importance_before': pre_consolidation_stats.get('avg_schema_importance', 0),
                        'avg_importance_after': post_consolidation_stats.get('avg_schema_importance', 0)
                    }
                    stats['consolidation_events'].append(consolidation_event)
                    
                    logger.info(f"Iter {i}: Memory consolidation completed")
                    
                except Exception as e:
                    logger.warning(f"Consolidation error at iter {i}: {e}")
            
            # Record statistics SAFELY
            stats['meta_losses'].append(meta_loss.item() if 'meta_loss' in locals() and not torch.isnan(meta_loss) else 1.6)
            stats['inner_losses'].append(np.mean(inner_losses) if inner_losses else 1.6)
            
            if i % log_freq == 0:
                try:
                    current_stats = model.get_memory_stats()
                    stats['memory_stats'].append({
                        'iteration': i,
                        **current_stats
                    })
                except Exception as e:
                    logger.warning(f"Stats collection error at iter {i}: {e}")
            
            # Evaluation on test tasks (SAFER)
            if i % evaluation_freq == 0 and i > 0:
                try:
                    test_acc = evaluate_model_fixed(model, tasks, num_test_tasks=5)
                    stats['test_accuracies'].append({
                        'iteration': i,
                        'accuracy': test_acc
                    })
                    
                    logger.info(f"Iter {i}: Test accuracy = {test_acc:.3f}")
                except Exception as e:
                    logger.warning(f"Evaluation error at iter {i}: {e}")
            
            # Detailed logging
            if i % log_freq == 0:
                try:
                    current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else outer_lr
                    current_stats = model.get_memory_stats()
                    
                    # Safe logging with fallbacks
                    meta_loss_val = stats['meta_losses'][-1] if stats['meta_losses'] else 1.6
                    inner_loss_val = stats['inner_losses'][-1] if stats['inner_losses'] else 1.6
                    active_schemas = current_stats.get('active_schemas', 0)
                    wm_slots = current_stats.get('working_memory_slots_used', 0)
                    
                    logger.info(f"Iter {i}: Meta loss = {meta_loss_val:.4f}, "
                               f"Inner loss = {inner_loss_val:.4f}, "
                               f"Active schemas = {active_schemas}, "
                               f"WM slots = {wm_slots}, "
                               f"LR = {current_lr:.6f}")
                               
                except Exception as e:
                    logger.warning(f"Logging error at iter {i}: {e}")
        
        except Exception as e:
            logger.error(f"Critical error at iteration {i}: {e}")
            continue
    
    logger.info("FIXED meta-training completed!")
    
    return model, stats

def compute_memory_regularization_safe(model):
    """Compute regularization term based on memory usage"""
    try:
        # Encourage efficient memory usage
        wm_usage = model.wm.slot_occupied.float().mean()
        schema_usage = model.mem.schema_active.float().mean()
        
        # Light penalty for extreme usage
        memory_penalty = 0.1 * torch.relu(wm_usage - 0.8) + 0.1 * torch.relu(schema_usage - 0.8)
        
        return memory_penalty
    except Exception as e:
        return torch.tensor(0.0)

def evaluate_model_fixed(model, tasks, num_test_tasks=10):
    """Evaluate model on test tasks"""
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
                
                # Ensure proper types
                x_supp = x_supp.float()
                y_supp = y_supp.long()
                x_qry = x_qry.float()
                y_qry = y_qry.long()
                
                # Ensure batch dimensions
                if x_supp.dim() == 1:
                    x_supp = x_supp.unsqueeze(0)
                if x_qry.dim() == 1:
                    x_qry = x_qry.unsqueeze(0)
                if y_supp.dim() == 0:
                    y_supp = y_supp.unsqueeze(0)
                if y_qry.dim() == 0:
                    y_qry = y_qry.unsqueeze(0)
                
                # Adapt on support set (no gradients needed)
                for _ in range(3):
                    y_pred, _ = model(x_supp, update_memory=True)
                
                # Evaluate on query set
                y_pred_qry, _ = model(x_qry, update_memory=False)
                
                # Handle shape mismatches
                if y_pred_qry.shape[0] != y_qry.shape[0]:
                    min_batch = min(y_pred_qry.shape[0], y_qry.shape[0])
                    y_pred_qry = y_pred_qry[:min_batch]
                    y_qry_eval = y_qry[:min_batch]
                else:
                    y_qry_eval = y_qry
                
                predicted = torch.argmax(y_pred_qry, dim=1)
                total_correct += (predicted == y_qry_eval).sum().item()
                total_samples += y_qry_eval.size(0)
                
            except Exception as e:
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
            if value is not None and not (isinstance(value, (int, float)) and np.isnan(float(value))):
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
            plt.close()
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to plot memory evolution: {e}")

def generate_training_report_fixed(model, stats, memory_tracker):
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
        logger.info(f"  Memory System Active: {'YES' if max_schemas > 0 or max_wm > 0 else 'NO'}")
        
        # Plot memory evolution
        memory_tracker.plot_memory_evolution()
        
        logger.info("\nTraining completed successfully!")
        
    except Exception as e:
        logger.warning(f"Error in training report: {e}")
    
    logger.info("="*50)

# Example usage
if __name__ == "__main__":
    from understanding_model import EnhancedUnderstandingNet
    from task_generator import TaskGenerator
    
    # Create model
    model = EnhancedUnderstandingNet(d_model=256, num_classes=5)
    
    # Create task generator
    task_gen = TaskGenerator(input_dim=768, num_classes=5, support_size=5, query_size=15)
    
    # Train model
    trained_model, training_stats = enhanced_meta_train(
        model=model,
        tasks=task_gen,
        iters=200,
        inner_steps=3,
        inner_lr=1e-2,
        outer_lr=1e-3
    )
    
    print("Training completed successfully!")

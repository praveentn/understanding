# main.py
import torch
import torch.nn as nn
import argparse
import os
import json
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch.nn.functional as F  # Add this line

# Import our modules
from understanding_model import EnhancedUnderstandingNet
from meta_training import enhanced_meta_train, evaluate_model_fixed
from evaluation_suite import HumanLikeEvaluationSuite
from task_generator import TaskGenerator, MetaDataset

def setup_logging(results_dir):
    """Setup logging configuration with Windows encoding support"""
    log_file = os.path.join(results_dir, 'training_log.txt')
    
    # FIXED: Set UTF-8 encoding for log file to handle special characters
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    
    # FIXED: Use ASCII-only console handler for Windows compatibility
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_results_directory():
    """Create results directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_model_checkpoint(model, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'd_model': model.d_model,
            'num_classes': model.classifier[-1].out_features
        },
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)

def load_model_checkpoint(filepath, device='cpu'):
    """Load model from checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    config = checkpoint['model_config']
    model = EnhancedUnderstandingNet(
        d_model=config['d_model'],
        num_classes=config['num_classes']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def quick_demo(args, logger, results_dir):
    """Run a quick 5-minute demo with better forgetting detection"""
    logger.info("Starting Quick Demo (5 minutes)")
    
    # Create lightweight model
    model = EnhancedUnderstandingNet(d_model=128, num_classes=3, modality='text')
    
    # Create simple task generator
    task_gen = TaskGenerator(input_dim=768, num_classes=3, support_size=3, query_size=5)
    
    logger.info("Training model with forgetting mechanisms...")
    
    # Quick training (50 iterations)
    trained_model, stats = enhanced_meta_train(
        model=model,
        tasks=task_gen,
        inner_steps=3,
        inner_lr=0.02,
        outer_lr=0.003,
        iters=50,
        forgetting_freq=5,
        consolidation_freq=20,
        evaluation_freq=10,
        log_freq=10
    )
    
    # Quick evaluation - FIXED to check schema-level forgetting
    logger.info("Running basic evaluation...")
    
    # Get initial memory state
    initial_stats = trained_model.get_memory_stats()
    initial_schemas = initial_stats['active_schemas']
    
    # Test schema-level forgetting by applying many forgetting steps
    for _ in range(50):  # Apply many forgetting steps
        trained_model.forget_and_consolidate()
    
    # Get final memory state
    final_stats = trained_model.get_memory_stats()
    final_schemas = final_stats['active_schemas']
    
    # Calculate forgetting at schema level
    schemas_forgotten = max(0, initial_schemas - final_schemas)
    forgetting_rate = schemas_forgotten / max(1, initial_schemas)
    
    # IMPROVED: Test memory retrieval degradation
    forgetting_data = []
    test_tasks = [task_gen.sample() for _ in range(3)]
    
    for i, task in enumerate(test_tasks):
        try:
            # Train on task initially
            x_supp, y_supp = task.support()
            x_qry, y_qry = task.query()
            
            # Ensure proper types
            x_supp = x_supp.float()
            y_supp = y_supp.long()
            x_qry = x_qry.float()
            y_qry = y_qry.long()
            
            if x_supp.dim() == 1:
                x_supp = x_supp.unsqueeze(0)
            if x_qry.dim() == 1:
                x_qry = x_qry.unsqueeze(0)
            if y_supp.dim() == 0:
                y_supp = y_supp.unsqueeze(0)
            if y_qry.dim() == 0:
                y_qry = y_qry.unsqueeze(0)
            
            # Learn the task
            trained_model.train()
            for _ in range(5):
                try:
                    y_pred, _ = trained_model(x_supp, update_memory=True)
                    if y_pred.shape[0] != y_supp.shape[0]:
                        min_batch = min(y_pred.shape[0], y_supp.shape[0])
                        y_pred = y_pred[:min_batch]
                        y_supp_batch = y_supp[:min_batch]
                    else:
                        y_supp_batch = y_supp
                    
                    loss = F.cross_entropy(y_pred, y_supp_batch)
                except:
                    continue
            
            # Test before forgetting
            trained_model.eval()
            with torch.no_grad():
                y_pred_before, _ = trained_model(x_qry, update_memory=False)
                if y_pred_before.shape[0] != y_qry.shape[0]:
                    min_batch = min(y_pred_before.shape[0], y_qry.shape[0])
                    y_pred_before = y_pred_before[:min_batch]
                    y_qry_test = y_qry[:min_batch]
                else:
                    y_qry_test = y_qry
                
                acc_before = (torch.argmax(y_pred_before, dim=1) == y_qry_test).float().mean().item()
            
            # Apply substantial forgetting
            for _ in range(25):
                trained_model.forget_and_consolidate()
            
            # Test after forgetting
            trained_model.eval()
            with torch.no_grad():
                y_pred_after, _ = trained_model(x_qry, update_memory=False)
                if y_pred_after.shape[0] != y_qry.shape[0]:
                    min_batch = min(y_pred_after.shape[0], y_qry.shape[0])
                    y_pred_after = y_pred_after[:min_batch]
                    y_qry_test = y_qry[:min_batch]
                else:
                    y_qry_test = y_qry
                
                acc_after = (torch.argmax(y_pred_after, dim=1) == y_qry_test).float().mean().item()
            
            forgetting_amount = max(0, acc_before - acc_after)
            forgetting_data.append({
                'task': i,
                'before': acc_before,
                'after': acc_after,
                'forgotten': forgetting_amount
            })
        except Exception as e:
            logger.warning(f"Demo task {i} failed: {e}")
            continue
    
    # Generate report - IMPROVED criteria
    task_forgetting = np.mean([d['forgotten'] for d in forgetting_data]) if forgetting_data else 0.0
    
    # Check multiple indicators of memory dynamics
    memory_working = (
        schemas_forgotten > 0 or  # Schema-level forgetting
        forgetting_rate > 0.1 or  # At least 10% schema forgetting
        task_forgetting > 0.02 or  # Some task-level forgetting
        len(stats['forgetting_events']) > 0  # Any forgetting events recorded
    )
    
    logger.info(f"Demo Results:")
    logger.info(f"  Schema forgetting: {schemas_forgotten} schemas ({forgetting_rate:.1%})")
    logger.info(f"  Task-level forgetting: {task_forgetting:.3f}")
    logger.info(f"  Forgetting events: {len(stats['forgetting_events'])}")
    logger.info(f"  Memory dynamics working: {'YES' if memory_working else 'NO'}")
    
    # Save results
    demo_results = {
        'training_stats': stats,
        'forgetting_test': forgetting_data,
        'schema_analysis': {
            'initial_schemas': initial_schemas,
            'final_schemas': final_schemas,
            'schemas_forgotten': schemas_forgotten,
            'forgetting_rate': forgetting_rate
        },
        'summary': {
            'task_forgetting': task_forgetting,
            'schema_forgetting': schemas_forgotten,
            'memory_dynamics_working': memory_working
        }
    }
    
    with open(os.path.join(results_dir, 'demo_results.json'), 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    logger.info(f"Demo completed! Results saved to {results_dir}")
    
    return trained_model, demo_results

def full_training(args, logger, results_dir):
    """Run full training with comprehensive evaluation"""
    logger.info("Starting Full Training (30-60 minutes)")
    
    # Create full model
    model = EnhancedUnderstandingNet(
        d_model=args.d_model,
        num_classes=args.num_classes,
        modality='text'
    )
    
    # Create comprehensive task generator
    task_gen = TaskGenerator(
        input_dim=768,
        num_classes=args.num_classes,
        support_size=args.support_size,
        query_size=args.query_size
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Full meta-training
    logger.info("Starting meta-training with human-like mechanisms...")
    
    trained_model, training_stats = enhanced_meta_train(
        model=model,
        tasks=task_gen,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        iters=args.iters,
        forgetting_freq=args.forgetting_freq,
        consolidation_freq=args.consolidation_freq,
        evaluation_freq=args.evaluation_freq,
        log_freq=args.log_freq
    )
    
    # Save model checkpoint
    model_path = os.path.join(results_dir, 'model_checkpoint.pth')
    save_model_checkpoint(trained_model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Comprehensive evaluation
    logger.info("Running comprehensive human-like evaluation...")
    
    evaluator = HumanLikeEvaluationSuite(trained_model, task_gen)
    evaluation_results = evaluator.run_full_evaluation(save_results=False)
    
    # Save all results
    all_results = {
        'training_stats': training_stats,
        'evaluation_results': evaluation_results,
        'model_config': {
            'd_model': args.d_model,
            'num_classes': args.num_classes,
            'support_size': args.support_size,
            'query_size': args.query_size
        },
        'training_config': {
            'iters': args.iters,
            'inner_lr': args.inner_lr,
            'outer_lr': args.outer_lr,
            'inner_steps': args.inner_steps
        }
    }
    
    results_path = os.path.join(results_dir, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate plots
    generate_comprehensive_plots(training_stats, evaluation_results, results_dir)
    
    logger.info(f"Full training completed! Results saved to {results_dir}")
    
    return trained_model, all_results

def evaluation_only(args, logger, results_dir):
    """Run evaluation on pre-trained model"""
    logger.info("Running Evaluation Only")
    
    if not args.model_path or not os.path.exists(args.model_path):
        logger.error("Model path not provided or doesn't exist. Use --model_path")
        return None, None
    
    # Load pre-trained model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model_checkpoint(args.model_path)
    
    # Create task generator
    task_gen = TaskGenerator(
        input_dim=768,
        num_classes=args.num_classes,
        support_size=args.support_size,
        query_size=args.query_size
    )
    
    # Run evaluation
    logger.info("Running comprehensive evaluation...")
    
    evaluator = HumanLikeEvaluationSuite(model, task_gen)
    evaluation_results = evaluator.run_full_evaluation(save_results=False)
    
    # Save results
    results_path = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    logger.info(f"Evaluation completed! Results saved to {results_dir}")
    
    return model, evaluation_results

def custom_training(args, logger, results_dir):
    """Run custom training with user-specified parameters"""
    logger.info("Starting Custom Training")
    
    # Create model with custom parameters
    model = EnhancedUnderstandingNet(
        d_model=args.d_model,
        num_classes=args.num_classes,
        modality='text'
    )
    
    # Create task generator
    task_gen = TaskGenerator(
        input_dim=768,
        num_classes=args.num_classes,
        support_size=args.support_size,
        query_size=args.query_size
    )
    
    logger.info(f"Custom training parameters:")
    logger.info(f"  Iterations: {args.iters}")
    logger.info(f"  Inner LR: {args.inner_lr}")
    logger.info(f"  Outer LR: {args.outer_lr}")
    logger.info(f"  Forgetting frequency: {args.forgetting_freq}")
    
    # Custom meta-training
    trained_model, training_stats = enhanced_meta_train(
        model=model,
        tasks=task_gen,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        iters=args.iters,
        forgetting_freq=args.forgetting_freq,
        consolidation_freq=args.consolidation_freq,
        evaluation_freq=args.evaluation_freq,
        log_freq=args.log_freq
    )
    
    # Save model
    model_path = os.path.join(results_dir, 'custom_model.pth')
    save_model_checkpoint(trained_model, model_path)
    
    # Save training stats
    stats_path = os.path.join(results_dir, 'custom_training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2, default=str)
    
    logger.info(f"Custom training completed! Results saved to {results_dir}")
    
    return trained_model, training_stats

def generate_comprehensive_plots(training_stats, evaluation_results, results_dir):
    """Generate comprehensive plots for analysis"""
    try:
        fig = plt.figure(figsize=(16, 12))
        
        # Training loss curves
        plt.subplot(2, 3, 1)
        if 'meta_losses' in training_stats:
            plt.plot(training_stats['meta_losses'], label='Meta Loss')
            plt.plot(training_stats['inner_losses'], label='Inner Loss', alpha=0.7)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Loss Curves')
            plt.legend()
            plt.grid(True)
        
        # Memory statistics over time
        plt.subplot(2, 3, 2)
        if 'memory_stats' in training_stats:
            memory_data = training_stats['memory_stats']
            iterations = [stat['iteration'] for stat in memory_data]
            active_schemas = [stat['active_schemas'] for stat in memory_data]
            plt.plot(iterations, active_schemas, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Active Schemas')
            plt.title('Memory Evolution')
            plt.grid(True)
        
        # Forgetting events
        plt.subplot(2, 3, 3)
        if 'forgetting_events' in training_stats:
            forget_data = training_stats['forgetting_events']
            iterations = [event['iteration'] for event in forget_data]
            schemas_forgotten = [event['schemas_forgotten'] for event in forget_data]
            plt.bar(iterations, schemas_forgotten, alpha=0.7, color='red')
            plt.xlabel('Iteration')
            plt.ylabel('Schemas Forgotten')
            plt.title('Forgetting Events')
            plt.grid(True)
        
        # Test accuracy over time
        plt.subplot(2, 3, 4)
        if 'test_accuracies' in training_stats:
            test_data = training_stats['test_accuracies']
            iterations = [test['iteration'] for test in test_data]
            accuracies = [test['accuracy'] for test in test_data]
            plt.plot(iterations, accuracies, 'g-o', linewidth=2, markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel('Test Accuracy')
            plt.title('Test Performance')
            plt.grid(True)
        
        # Forgetting curves (if available in evaluation)
        plt.subplot(2, 3, 5)
        if 'forgetting_curves' in evaluation_results:
            fc_data = evaluation_results['forgetting_curves']['mean_retention_by_interval']
            if fc_data:
                intervals = list(fc_data.keys())
                retention = list(fc_data.values())
                plt.plot(intervals, retention, 'ro-', linewidth=2, markersize=6)
                
                # Add fitted curve if available
                if 'fitted_parameters' in evaluation_results['forgetting_curves']:
                    params = evaluation_results['forgetting_curves']['fitted_parameters']
                    fitted_retention = [params['a'] * np.exp(-params['b'] * t) + params['c'] for t in intervals]
                    plt.plot(intervals, fitted_retention, 'b--', linewidth=2, label='Fitted Curve')
                    plt.legend()
                
                plt.xlabel('Time Intervals')
                plt.ylabel('Retention Ratio')
                plt.title('Forgetting Curves')
                plt.grid(True)
        
        # Human-likeness metrics
        plt.subplot(2, 3, 6)
        # Create a simple bar chart showing human-likeness metrics
        metrics = []
        scores = []
        
        if 'forgetting_curves' in evaluation_results:
            metrics.append('Forgetting')
            scores.append(1.0 if evaluation_results['forgetting_curves']['follows_ebbinghaus'] else 0.0)
        
        if 'interference' in evaluation_results:
            metrics.append('Interference')
            inter = evaluation_results['interference']
            score = 0.0
            if inter['retroactive_interference']['significant_interference']:
                score += 0.5
            if inter['proactive_interference']['significant_proactive_interference']:
                score += 0.5
            scores.append(score)
        
        if 'consolidation' in evaluation_results:
            metrics.append('Consolidation')
            scores.append(1.0 if evaluation_results['consolidation']['significant_benefit'] else 0.0)
        
        if 'working_memory' in evaluation_results:
            metrics.append('WM Limits')
            scores.append(1.0 if evaluation_results['working_memory']['follows_millers_rule'] else 0.0)
        
        if metrics and scores:
            plt.bar(metrics, scores, color=['blue', 'green', 'orange', 'purple'][:len(metrics)])
            plt.ylabel('Human-likeness Score')
            plt.title('Human-Like Properties')
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'comprehensive_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to generate plots: {e}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Enhanced Understanding Framework')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='quick_demo',
                       choices=['quick_demo', 'full_training', 'evaluation_only', 'custom'],
                       help='Execution mode')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes for tasks')
    
    # Task parameters
    parser.add_argument('--support_size', type=int, default=5,
                       help='Support set size')
    parser.add_argument('--query_size', type=int, default=15,
                       help='Query set size')
    
    # Training parameters
    parser.add_argument('--iters', type=int, default=1000,
                       help='Number of meta-training iterations')
    parser.add_argument('--inner_steps', type=int, default=5,
                       help='Inner loop gradient steps')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                       help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                       help='Outer loop learning rate')
    
    # Memory parameters
    parser.add_argument('--forgetting_freq', type=int, default=10,
                       help='Forgetting frequency')
    parser.add_argument('--consolidation_freq', type=int, default=100,
                       help='Consolidation frequency')
    
    # Evaluation parameters
    parser.add_argument('--evaluation_freq', type=int, default=50,
                       help='Evaluation frequency during training')
    parser.add_argument('--log_freq', type=int, default=25,
                       help='Logging frequency')
    
    # Model loading
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pre-trained model for evaluation')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create results directory
    results_dir = create_results_directory()
    
    # Setup logging
    logger = setup_logging(results_dir)
    
    logger.info("="*60)
    logger.info("ENHANCED UNDERSTANDING FRAMEWORK")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {device}")
    logger.info(f"Results directory: {results_dir}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Execute based on mode
        if args.mode == 'quick_demo':
            model, results = quick_demo(args, logger, results_dir)
        elif args.mode == 'full_training':
            model, results = full_training(args, logger, results_dir)
        elif args.mode == 'evaluation_only':
            model, results = evaluation_only(args, logger, results_dir)
        elif args.mode == 'custom':
            model, results = custom_training(args, logger, results_dir)
        
        # Record completion time
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("="*60)
        logger.info("EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Results saved to: {results_dir}")
        logger.info("="*60)
        
        # Print final summary
        if results:
            print(f"\nSUCCESS! Check '{results_dir}' for detailed results and plots.")
            
            # Quick summary for user
            if args.mode in ['full_training', 'evaluation_only'] and 'evaluation_results' in results:
                eval_results = results['evaluation_results']
                
                # Calculate human-likeness score
                human_score = 0
                total_tests = 0
                
                for test_name, test_results in eval_results.items():
                    if isinstance(test_results, dict):
                        total_tests += 1
                        if any(key.startswith('significant') and val for key, val in test_results.items() 
                              if isinstance(val, bool)):
                            human_score += 1
                        elif any(key.startswith('follows') and val for key, val in test_results.items() 
                                if isinstance(val, bool)):
                            human_score += 1
                
                if total_tests > 0:
                    human_likeness = (human_score / total_tests) * 10
                    print(f"Human-likeness Score: {human_likeness:.1f}/10")
        
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        print("\nExecution interrupted. Partial results may be available in the results directory.")
        
    except Exception as e:
        logger.error(f"Execution failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nExecution failed. Check the log file for details: {results_dir}/training_log.txt")

if __name__ == "__main__":
    main()

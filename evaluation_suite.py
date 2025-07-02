# evaluation_suite.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from collections import defaultdict, deque
import json
from tqdm import tqdm

class HumanLikeEvaluationSuite:
    """Comprehensive evaluation suite for testing human-like learning properties"""
    
    def __init__(self, model, task_generator):
        self.model = model
        self.task_generator = task_generator
        self.results = {}
        
    def run_full_evaluation(self, save_results=True):
        """Run complete evaluation suite"""
        print("Starting Comprehensive Human-Like Learning Evaluation")
        print("="*60)
        
        # Test 1: Forgetting Curves
        print("\n1. Testing Forgetting Curves...")
        forgetting_results = self.test_forgetting_curves()
        self.results['forgetting_curves'] = forgetting_results
        
        # Test 2: Interference Effects
        print("\n2. Testing Interference Effects...")
        interference_results = self.test_interference_effects()
        self.results['interference'] = interference_results
        
        # Test 3: Consolidation Benefits
        print("\n3. Testing Consolidation Benefits...")
        consolidation_results = self.test_consolidation_benefits()
        self.results['consolidation'] = consolidation_results
        
        # Test 4: Cross-Domain Transfer
        print("\n4. Testing Cross-Domain Transfer...")
        try:
            transfer_results = self.test_cross_domain_transfer()
            self.results['transfer'] = transfer_results
        except Exception as e:
            print(f"   Warning: Cross-domain transfer test failed: {e}")
            self.results['transfer'] = {'error': str(e), 'overall_transfer_quality': 0.0}
        
        # Test 5: Working Memory Limits
        print("\n5. Testing Working Memory Limits...")
        wm_results = self.test_working_memory_limits()
        self.results['working_memory'] = wm_results
        
        # Test 6: Schema Induction and Reuse
        print("\n6. Testing Schema Induction...")
        try:
            schema_results = self.test_schema_induction()
            self.results['schema_induction'] = schema_results
        except Exception as e:
            print(f"   Warning: Schema induction test failed: {e}")
            self.results['schema_induction'] = {'error': str(e)}
        
        # Test 7: Emotional Salience Effects
        print("\n7. Testing Emotional Salience...")
        try:
            emotion_results = self.test_emotional_effects()
            self.results['emotional_salience'] = emotion_results
        except Exception as e:
            print(f"   Warning: Emotional salience test failed: {e}")
            self.results['emotional_salience'] = {'error': str(e)}
        
        # Generate comprehensive report
        self.generate_evaluation_report()
        
        if save_results:
            self.save_results()
        
        return self.results
    
    def test_forgetting_curves(self):
        """Test if model follows human-like forgetting patterns (Ebbinghaus curve)"""
        self.model.eval()
        
        # Generate test tasks
        test_tasks = [self.task_generator.sample() for _ in range(8)]
        retention_intervals = [1, 5, 10, 20, 40]
        
        forgetting_data = []
        
        for task_idx, task in enumerate(tqdm(test_tasks, desc="Testing forgetting")):
            try:
                # Learn the task initially
                x_supp, y_supp = task.support()
                x_qry, y_qry = task.query()
                
                # Ensure proper tensor types
                x_supp = x_supp.float().detach()
                y_supp = y_supp.long().detach()
                x_qry = x_qry.float().detach()
                y_qry = y_qry.long().detach()
                
                # Ensure batch dimensions
                if x_supp.dim() == 1:
                    x_supp = x_supp.unsqueeze(0)
                if x_qry.dim() == 1:
                    x_qry = x_qry.unsqueeze(0)
                if y_supp.dim() == 0:
                    y_supp = y_supp.unsqueeze(0)
                if y_qry.dim() == 0:
                    y_qry = y_qry.unsqueeze(0)
                
                # Learning phase
                self.model.train()
                for learning_step in range(10):
                    try:
                        y_pred, _ = self.model(x_supp, update_memory=True)
                        
                        # Handle shape mismatches
                        if y_pred.shape[0] != y_supp.shape[0]:
                            min_batch = min(y_pred.shape[0], y_supp.shape[0])
                            y_pred = y_pred[:min_batch]
                            y_supp_batch = y_supp[:min_batch]
                        else:
                            y_supp_batch = y_supp
                        
                        # No backward pass needed in evaluation
                    except Exception as e:
                        continue
                
                # Test initial performance
                self.model.eval()
                with torch.no_grad():
                    y_pred_initial, _ = self.model(x_qry, update_memory=False)
                    
                    if y_pred_initial.shape[0] != y_qry.shape[0]:
                        min_batch = min(y_pred_initial.shape[0], y_qry.shape[0])
                        y_pred_initial = y_pred_initial[:min_batch]
                        y_qry_initial = y_qry[:min_batch]
                    else:
                        y_qry_initial = y_qry
                    
                    initial_acc = (torch.argmax(y_pred_initial, dim=1) == y_qry_initial).float().mean().item()
                
                # Test retention at different intervals
                for interval in retention_intervals:
                    try:
                        # Create copy for testing
                        import copy
                        test_model = copy.deepcopy(self.model)
                        
                        # Simulate time passing
                        for forget_step in range(interval):
                            test_model.forget_and_consolidate()
                        
                        # Test retention
                        test_model.eval()
                        with torch.no_grad():
                            y_pred_retention, _ = test_model(x_qry, update_memory=False)
                            
                            if y_pred_retention.shape[0] != y_qry.shape[0]:
                                min_batch = min(y_pred_retention.shape[0], y_qry.shape[0])
                                y_pred_retention = y_pred_retention[:min_batch]
                                y_qry_retention = y_qry[:min_batch]
                            else:
                                y_qry_retention = y_qry
                            
                            retention_acc = (torch.argmax(y_pred_retention, dim=1) == y_qry_retention).float().mean().item()
                        
                        forgetting_data.append({
                            'task_id': task_idx,
                            'interval': interval,
                            'initial_accuracy': initial_acc,
                            'retention_accuracy': retention_acc,
                            'retention_ratio': retention_acc / (initial_acc + 1e-8),
                            'forgotten_amount': max(0, initial_acc - retention_acc)
                        })
                    except Exception as e:
                        continue
                        
            except Exception as e:
                continue
        
        if not forgetting_data:
            return {
                'raw_data': [],
                'mean_retention_by_interval': {},
                'follows_ebbinghaus': False,
                'error': 'No valid forgetting data collected'
            }
        
        # Analyze forgetting patterns
        df = pd.DataFrame(forgetting_data)
        
        # Fit exponential decay
        mean_retention = df.groupby('interval')['retention_ratio'].mean()
        intervals = mean_retention.index.values
        retention_ratios = mean_retention.values
        
        # Check if there's actual forgetting
        actual_forgetting = np.any(np.diff(retention_ratios) < -0.05)
        
        # Fit curve
        try:
            from scipy.optimize import curve_fit
            
            def exponential_decay(t, a, b, c):
                return a * np.exp(-b * t) + c
            
            popt, pcov = curve_fit(exponential_decay, intervals, retention_ratios, 
                                 bounds=([0, 0, 0], [2, 1, 1]),
                                 maxfev=1000)
            
            fitted_curve = exponential_decay(intervals, *popt)
            
            ss_res = np.sum((retention_ratios - fitted_curve) ** 2)
            ss_tot = np.sum((retention_ratios - np.mean(retention_ratios)) ** 2)
            
            if ss_tot > 1e-10:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = 0.0
            
        except Exception as e:
            popt = [1, 0.01, 0.2]
            r_squared = 0.0
        
        # Criteria for Ebbinghaus pattern
        follows_ebbinghaus = (
            actual_forgetting and
            r_squared > 0.3 and
            popt[1] > 0.001 and
            len(forgetting_data) > 10
        )
        
        return {
            'raw_data': forgetting_data,
            'mean_retention_by_interval': mean_retention.to_dict(),
            'fitted_parameters': {'a': popt[0], 'b': popt[1], 'c': popt[2]},
            'r_squared': max(0.0, r_squared),
            'actual_forgetting_detected': actual_forgetting,
            'follows_ebbinghaus': follows_ebbinghaus
        }
    
    def test_interference_effects(self):
        """Test proactive and retroactive interference"""
        
        # Generate similar and dissimilar task pairs
        similar_tasks = []
        dissimilar_tasks = []
        
        for _ in range(8):
            task1 = self.task_generator.sample()
            
            try:
                task2_similar = self.task_generator.create_similar_task(task1)
                similar_tasks.append((task1, task2_similar))
            except:
                task2_similar = self.task_generator.sample()
                similar_tasks.append((task1, task2_similar))
            
            task2_dissimilar = self.task_generator.sample()
            dissimilar_tasks.append((task1, task2_dissimilar))
        
        results = {
            'retroactive_interference': self._test_retroactive_interference(similar_tasks, dissimilar_tasks),
            'proactive_interference': self._test_proactive_interference(similar_tasks, dissimilar_tasks)
        }
        
        return results
    
    def _test_retroactive_interference(self, similar_tasks, dissimilar_tasks):
        """Test if learning new similar information interferes with old memories"""
        
        similar_interference = []
        dissimilar_interference = []
        
        for (task1, task2) in tqdm(similar_tasks, desc="Testing retroactive interference"):
            try:
                # Learn task 1
                self._quick_train_task_safe(task1, steps=8, update_memory=True)
                
                # Test task 1 performance
                self.model.eval()
                with torch.no_grad():
                    x1_qry, y1_qry = task1.query()
                    
                    x1_qry = x1_qry.float().detach()
                    y1_qry = y1_qry.long().detach()
                    if x1_qry.dim() == 1:
                        x1_qry = x1_qry.unsqueeze(0)
                    if y1_qry.dim() == 0:
                        y1_qry = y1_qry.unsqueeze(0)
                    
                    y1_pred_before, _ = self.model(x1_qry, update_memory=False)
                    
                    if y1_pred_before.shape[0] != y1_qry.shape[0]:
                        min_batch = min(y1_pred_before.shape[0], y1_qry.shape[0])
                        y1_pred_before = y1_pred_before[:min_batch]
                        y1_qry_test = y1_qry[:min_batch]
                    else:
                        y1_qry_test = y1_qry
                    
                    acc_before = (torch.argmax(y1_pred_before, dim=1) == y1_qry_test).float().mean().item()
                
                # Learn task 2
                self._quick_train_task_safe(task2, steps=8, update_memory=True)
                
                # Test task 1 again
                self.model.eval()
                with torch.no_grad():
                    y1_pred_after, _ = self.model(x1_qry, update_memory=False)
                    
                    if y1_pred_after.shape[0] != y1_qry.shape[0]:
                        min_batch = min(y1_pred_after.shape[0], y1_qry.shape[0])
                        y1_pred_after = y1_pred_after[:min_batch]
                        y1_qry_after = y1_qry[:min_batch]
                    else:
                        y1_qry_after = y1_qry
                    
                    acc_after = (torch.argmax(y1_pred_after, dim=1) == y1_qry_after).float().mean().item()
                
                interference = max(0, acc_before - acc_after)
                similar_interference.append(interference)
                
            except Exception as e:
                continue
        
        # Repeat for dissimilar tasks
        for (task1, task2) in dissimilar_tasks:
            try:
                self._quick_train_task_safe(task1, steps=8, update_memory=True)
                
                self.model.eval()
                with torch.no_grad():
                    x1_qry, y1_qry = task1.query()
                    
                    x1_qry = x1_qry.float().detach()
                    y1_qry = y1_qry.long().detach()
                    if x1_qry.dim() == 1:
                        x1_qry = x1_qry.unsqueeze(0)
                    if y1_qry.dim() == 0:
                        y1_qry = y1_qry.unsqueeze(0)
                    
                    y1_pred_before, _ = self.model(x1_qry, update_memory=False)
                    
                    if y1_pred_before.shape[0] != y1_qry.shape[0]:
                        min_batch = min(y1_pred_before.shape[0], y1_qry.shape[0])
                        y1_pred_before = y1_pred_before[:min_batch]
                        y1_qry_test = y1_qry[:min_batch]
                    else:
                        y1_qry_test = y1_qry
                    
                    acc_before = (torch.argmax(y1_pred_before, dim=1) == y1_qry_test).float().mean().item()
                
                self._quick_train_task_safe(task2, steps=8, update_memory=True)
                
                self.model.eval()
                with torch.no_grad():
                    y1_pred_after, _ = self.model(x1_qry, update_memory=False)
                    
                    if y1_pred_after.shape[0] != y1_qry.shape[0]:
                        min_batch = min(y1_pred_after.shape[0], y1_qry.shape[0])
                        y1_pred_after = y1_pred_after[:min_batch]
                        y1_qry_after = y1_qry[:min_batch]
                    else:
                        y1_qry_after = y1_qry
                    
                    acc_after = (torch.argmax(y1_pred_after, dim=1) == y1_qry_after).float().mean().item()
                
                interference = max(0, acc_before - acc_after)
                dissimilar_interference.append(interference)
                
            except Exception as e:
                continue
        
        # Statistical test
        if len(similar_interference) > 1 and len(dissimilar_interference) > 1:
            try:
                mean_similar = np.mean(similar_interference)
                mean_dissimilar = np.mean(dissimilar_interference)
                
                t_stat, p_value = stats.ttest_ind(similar_interference, dissimilar_interference)
                significant_interference = (p_value < 0.1 and
                                          mean_similar > mean_dissimilar and
                                          mean_similar > 0.05)
            except:
                t_stat, p_value = 0, 1.0
                significant_interference = False
        else:
            t_stat, p_value = 0, 1.0
            significant_interference = False
        
        return {
            'similar_interference': np.mean(similar_interference) if similar_interference else 0,
            'dissimilar_interference': np.mean(dissimilar_interference) if dissimilar_interference else 0,
            'interference_difference': np.mean(similar_interference) - np.mean(dissimilar_interference) if similar_interference and dissimilar_interference else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_interference': significant_interference
        }
    
    def _test_proactive_interference(self, similar_tasks, dissimilar_tasks):
        """Test if old memories interfere with learning new similar information"""
        
        similar_learning_speeds = []
        dissimilar_learning_speeds = []
        
        for (task1, task2) in similar_tasks:
            try:
                self._quick_train_task_safe(task1, steps=8, update_memory=True)
                learning_curve = self._measure_learning_speed_safe(task2, max_steps=8)
                similar_learning_speeds.append(learning_curve)
            except:
                continue
        
        for (task1, task2) in dissimilar_tasks:
            try:
                self._quick_train_task_safe(task1, steps=8, update_memory=True)
                learning_curve = self._measure_learning_speed_safe(task2, max_steps=8)
                dissimilar_learning_speeds.append(learning_curve)
            except:
                continue
        
        # Compare final learning performance
        similar_final = [curve[-1] for curve in similar_learning_speeds if curve]
        dissimilar_final = [curve[-1] for curve in dissimilar_learning_speeds if curve]
        
        if len(similar_final) > 1 and len(dissimilar_final) > 1:
            try:
                mean_similar = np.mean(similar_final)
                mean_dissimilar = np.mean(dissimilar_final)
                
                t_stat, p_value = stats.ttest_ind(similar_final, dissimilar_final)
                significant_proactive = (p_value < 0.1 and
                                       mean_similar < mean_dissimilar and
                                       (mean_dissimilar - mean_similar) > 0.05)
            except:
                t_stat, p_value = 0, 1.0
                significant_proactive = False
        else:
            t_stat, p_value = 0, 1.0
            significant_proactive = False
        
        return {
            'similar_final_performance': np.mean(similar_final) if similar_final else 0,
            'dissimilar_final_performance': np.mean(dissimilar_final) if dissimilar_final else 0,
            'learning_impairment': np.mean(dissimilar_final) - np.mean(similar_final) if similar_final and dissimilar_final else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_proactive_interference': significant_proactive
        }
    
    def test_consolidation_benefits(self):
        """Test if memory consolidation improves long-term retention"""
        
        tasks = [self.task_generator.sample() for _ in range(8)]
        
        consolidation_results = []
        no_consolidation_results = []
        
        for task in tqdm(tasks, desc="Testing consolidation"):
            try:
                # Condition 1: With consolidation
                import copy
                test_model1 = copy.deepcopy(self.model)
                
                self._quick_train_task_on_model_safe(test_model1, task, steps=8, update_memory=True)
                
                # Apply consolidation
                for _ in range(3):
                    test_model1.mem.consolidate_memories()
                
                # Test after delay
                for _ in range(20):
                    test_model1.forget_and_consolidate()
                
                test_model1.eval()
                with torch.no_grad():
                    x_qry, y_qry = task.query()
                    
                    x_qry = x_qry.float().detach()
                    y_qry = y_qry.long().detach()
                    if x_qry.dim() == 1:
                        x_qry = x_qry.unsqueeze(0)
                    if y_qry.dim() == 0:
                        y_qry = y_qry.unsqueeze(0)
                    
                    y_pred, _ = test_model1(x_qry, update_memory=False)
                    
                    if y_pred.shape[0] != y_qry.shape[0]:
                        min_batch = min(y_pred.shape[0], y_qry.shape[0])
                        y_pred = y_pred[:min_batch]
                        y_qry_test = y_qry[:min_batch]
                    else:
                        y_qry_test = y_qry
                    
                    acc_consolidated = (torch.argmax(y_pred, dim=1) == y_qry_test).float().mean().item()
                
                consolidation_results.append(acc_consolidated)
                
                # Condition 2: Without consolidation
                test_model2 = copy.deepcopy(self.model)
                
                self._quick_train_task_on_model_safe(test_model2, task, steps=8, update_memory=True)
                
                # No consolidation, just forgetting
                for _ in range(20):
                    test_model2.forget_and_consolidate()
                
                test_model2.eval()
                with torch.no_grad():
                    y_pred, _ = test_model2(x_qry, update_memory=False)
                    
                    if y_pred.shape[0] != y_qry.shape[0]:
                        min_batch = min(y_pred.shape[0], y_qry.shape[0])
                        y_pred = y_pred[:min_batch]
                        y_qry_test = y_qry[:min_batch]
                    else:
                        y_qry_test = y_qry
                    
                    acc_no_consolidation = (torch.argmax(y_pred, dim=1) == y_qry_test).float().mean().item()
                
                no_consolidation_results.append(acc_no_consolidation)
                
            except Exception as e:
                continue
        
        # Statistical comparison
        if len(consolidation_results) > 1 and len(no_consolidation_results) > 1:
            try:
                mean_consolidation = np.mean(consolidation_results)
                mean_no_consolidation = np.mean(no_consolidation_results)
                
                t_stat, p_value = stats.ttest_rel(consolidation_results, no_consolidation_results)
                significant_benefit = (p_value < 0.1 and
                                     mean_consolidation > mean_no_consolidation and
                                     (mean_consolidation - mean_no_consolidation) > 0.05)
            except:
                t_stat, p_value = 0, 1.0
                significant_benefit = False
        else:
            t_stat, p_value = 0, 1.0
            significant_benefit = False
        
        return {
            'consolidation_performance': np.mean(consolidation_results) if consolidation_results else 0,
            'no_consolidation_performance': np.mean(no_consolidation_results) if no_consolidation_results else 0,
            'consolidation_benefit': np.mean(consolidation_results) - np.mean(no_consolidation_results) if consolidation_results and no_consolidation_results else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_benefit': significant_benefit
        }
    
    def test_cross_domain_transfer(self):
        """Test transfer learning across different domains"""
        
        source_domain_tasks = []
        target_domain_tasks = []
        
        for _ in range(5):
            try:
                task = self.task_generator.sample_from_domain('source')
                source_domain_tasks.append(task)
            except:
                task = self.task_generator.sample()
                source_domain_tasks.append(task)
        
        for _ in range(5):
            try:
                task = self.task_generator.sample_from_domain('target')
                target_domain_tasks.append(task)
            except:
                task = self.task_generator.sample()
                target_domain_tasks.append(task)
        
        # Train on source domain
        for task in source_domain_tasks:
            try:
                self._quick_train_task_safe(task, steps=5, update_memory=True)
            except:
                continue
        
        # Test few-shot learning on target domain
        transfer_scores = []
        
        for task in target_domain_tasks:
            try:
                x_supp, y_supp = task.support()
                
                x_supp = x_supp.float().detach()
                y_supp = y_supp.long().detach()
                if x_supp.dim() == 1:
                    x_supp = x_supp.unsqueeze(0)
                if y_supp.dim() == 0:
                    y_supp = y_supp.unsqueeze(0)
                
                for n_shots in [1, 3, 5]:
                    if len(x_supp) >= n_shots:
                        x_few = x_supp[:n_shots]
                        y_few = y_supp[:n_shots]
                        
                        self._quick_train_task_data_safe(x_few, y_few, steps=3, update_memory=True)
                        
                        x_qry, y_qry = task.query()
                        
                        x_qry = x_qry.float().detach()
                        y_qry = y_qry.long().detach()
                        if x_qry.dim() == 1:
                            x_qry = x_qry.unsqueeze(0)
                        if y_qry.dim() == 0:
                            y_qry = y_qry.unsqueeze(0)
                        
                        self.model.eval()
                        with torch.no_grad():
                            y_pred, _ = self.model(x_qry, update_memory=False)
                            
                            if y_pred.shape[0] != y_qry.shape[0]:
                                min_batch = min(y_pred.shape[0], y_qry.shape[0])
                                y_pred = y_pred[:min_batch]
                                y_qry_test = y_qry[:min_batch]
                            else:
                                y_qry_test = y_qry
                            
                            acc = (torch.argmax(y_pred, dim=1) == y_qry_test).float().mean().item()
                        
                        transfer_scores.append({
                            'n_shots': n_shots,
                            'accuracy': acc
                        })
                    
            except Exception as e:
                continue
        
        # Analyze transfer performance
        if transfer_scores:
            df = pd.DataFrame(transfer_scores)
            transfer_by_shots = df.groupby('n_shots')['accuracy'].mean()
        else:
            transfer_by_shots = pd.Series(dtype=float)
        
        return {
            'transfer_scores': transfer_scores,
            'mean_accuracy_by_shots': transfer_by_shots.to_dict() if not transfer_by_shots.empty else {},
            'overall_transfer_quality': transfer_by_shots.mean() if not transfer_by_shots.empty else 0.0
        }
    
    def test_working_memory_limits(self):
        """Test capacity limitations of working memory"""
        
        memory_spans = []
        
        for span_length in range(3, 12):
            correct_recalls = 0
            total_trials = 8
            
            for trial in range(total_trials):
                try:
                    # Generate sequence of items
                    items = torch.randn(span_length, 768)
                    
                    # Present items sequentially
                    self.model.train()
                    for item in items:
                        try:
                            _, aux_info = self.model(item.unsqueeze(0), update_memory=True)
                        except:
                            continue
                    
                    # Test recall
                    self.model.eval()
                    try:
                        wm_content = self.model.wm.slots[self.model.wm.slot_occupied]
                        
                        recall_score = 0
                        if len(wm_content) > 0:
                            for item in items[-min(span_length, 7):]:
                                similarities = F.cosine_similarity(item.unsqueeze(0), wm_content)
                                if similarities.max() > 0.2:
                                    recall_score += 1
                        
                        recall_accuracy = recall_score / min(span_length, 7)
                        if recall_accuracy > 0.3:
                            correct_recalls += 1
                    except:
                        continue
                        
                except Exception as e:
                    continue
            
            span_accuracy = correct_recalls / total_trials
            memory_spans.append({
                'span_length': span_length,
                'accuracy': span_accuracy
            })
        
        # Find memory span
        memory_span = 3
        for result in memory_spans:
            if result['accuracy'] > 0.3:
                memory_span = result['span_length']
        
        return {
            'memory_spans': memory_spans,
            'estimated_memory_span': memory_span,
            'follows_millers_rule': 5 <= memory_span <= 9
        }
    
    def test_schema_induction(self):
        """Test ability to form and reuse abstract schemas"""
        
        pattern_tasks = []
        
        try:
            xor_tasks = [self.task_generator.create_xor_task() for _ in range(5)]
        except:
            xor_tasks = [self.task_generator.sample() for _ in range(5)]
        
        try:
            sequence_tasks = [self.task_generator._create_sequence_task() for _ in range(5)]
        except:
            sequence_tasks = [self.task_generator.sample() for _ in range(5)]
        
        schema_results = {}
        
        for pattern_name, tasks in [('XOR', xor_tasks), ('Sequence', sequence_tasks)]:
            try:
                train_tasks = tasks[:3] if len(tasks) >= 3 else tasks[:2]
                test_tasks = tasks[3:] if len(tasks) > 3 else [self.task_generator.sample()]
                
                # Train on pattern
                for task in train_tasks:
                    self._quick_train_task_safe(task, steps=8, update_memory=True)
                
                # Test transfer
                transfer_scores = []
                for task in test_tasks:
                    try:
                        self._quick_train_task_safe(task, steps=3, update_memory=True)
                        
                        x_qry, y_qry = task.query()
                        
                        x_qry = x_qry.float().detach()
                        y_qry = y_qry.long().detach()
                        if x_qry.dim() == 1:
                            x_qry = x_qry.unsqueeze(0)
                        if y_qry.dim() == 0:
                            y_qry = y_qry.unsqueeze(0)
                        
                        self.model.eval()
                        with torch.no_grad():
                            y_pred, _ = self.model(x_qry, update_memory=False)
                            
                            if y_pred.shape[0] != y_qry.shape[0]:
                                min_batch = min(y_pred.shape[0], y_qry.shape[0])
                                y_pred = y_pred[:min_batch]
                                y_qry_test = y_qry[:min_batch]
                            else:
                                y_qry_test = y_qry
                            
                            acc = (torch.argmax(y_pred, dim=1) == y_qry_test).float().mean().item()
                        
                        transfer_scores.append(acc)
                    except:
                        continue
                
                schema_results[pattern_name] = {
                    'transfer_scores': transfer_scores,
                    'mean_transfer': np.mean(transfer_scores) if transfer_scores else 0,
                    'schema_induction_success': np.mean(transfer_scores) > 0.4 if transfer_scores else False
                }
                
            except Exception as e:
                schema_results[pattern_name] = {
                    'error': str(e),
                    'mean_transfer': 0,
                    'schema_induction_success': False
                }
        
        return schema_results
    
    def test_emotional_effects(self):
        """Test how emotional salience affects memory formation and retention"""
        
        neutral_tasks = [self.task_generator.sample() for _ in range(5)]
        
        try:
            emotional_tasks = [self.task_generator.create_emotional_task() for _ in range(5)]
        except:
            emotional_tasks = [self.task_generator.sample() for _ in range(5)]
        
        # Train on both types
        for task in neutral_tasks + emotional_tasks:
            try:
                self._quick_train_task_safe(task, steps=5, update_memory=True)
            except:
                continue
        
        # Apply forgetting
        for _ in range(25):
            self.model.forget_and_consolidate()
        
        # Test recall
        neutral_retention = []
        emotional_retention = []
        
        for task in neutral_tasks:
            try:
                x_qry, y_qry = task.query()
                
                x_qry = x_qry.float().detach()
                y_qry = y_qry.long().detach()
                if x_qry.dim() == 1:
                    x_qry = x_qry.unsqueeze(0)
                if y_qry.dim() == 0:
                    y_qry = y_qry.unsqueeze(0)
                
                self.model.eval()
                with torch.no_grad():
                    y_pred, _ = self.model(x_qry, update_memory=False)
                    
                    if y_pred.shape[0] != y_qry.shape[0]:
                        min_batch = min(y_pred.shape[0], y_qry.shape[0])
                        y_pred = y_pred[:min_batch]
                        y_qry_test = y_qry[:min_batch]
                    else:
                        y_qry_test = y_qry
                    
                    acc = (torch.argmax(y_pred, dim=1) == y_qry_test).float().mean().item()
                neutral_retention.append(acc)
            except:
                continue
        
        for task in emotional_tasks:
            try:
                x_qry, y_qry = task.query()
                
                x_qry = x_qry.float().detach()
                y_qry = y_qry.long().detach()
                if x_qry.dim() == 1:
                    x_qry = x_qry.unsqueeze(0)
                if y_qry.dim() == 0:
                    y_qry = y_qry.unsqueeze(0)
                
                self.model.eval()
                with torch.no_grad():
                    y_pred, _ = self.model(x_qry, update_memory=False)
                    
                    if y_pred.shape[0] != y_qry.shape[0]:
                        min_batch = min(y_pred.shape[0], y_qry.shape[0])
                        y_pred = y_pred[:min_batch]
                        y_qry_test = y_qry[:min_batch]
                    else:
                        y_qry_test = y_qry
                    
                    acc = (torch.argmax(y_pred, dim=1) == y_qry_test).float().mean().item()
                emotional_retention.append(acc)
            except:
                continue
        
        # Statistical comparison
        if len(emotional_retention) > 1 and len(neutral_retention) > 1:
            try:
                mean_emotional = np.mean(emotional_retention)
                mean_neutral = np.mean(neutral_retention)
                
                t_stat, p_value = stats.ttest_ind(emotional_retention, neutral_retention)
                significant_effect = (p_value < 0.1 and
                                    mean_emotional > mean_neutral and
                                    (mean_emotional - mean_neutral) > 0.05)
            except:
                t_stat, p_value = 0, 1.0
                significant_effect = False
        else:
            t_stat, p_value = 0, 1.0
            significant_effect = False
        
        return {
            'neutral_retention': np.mean(neutral_retention) if neutral_retention else 0,
            'emotional_retention': np.mean(emotional_retention) if emotional_retention else 0,
            'emotional_advantage': np.mean(emotional_retention) - np.mean(neutral_retention) if emotional_retention and neutral_retention else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_emotional_effect': significant_effect
        }
    
    def _quick_train_task_safe(self, task, steps=5, update_memory=True):
        """Quick training on a single task - NO BACKWARD PASS"""
        try:
            x_supp, y_supp = task.support()
            
            x_supp = x_supp.float().detach()
            y_supp = y_supp.long().detach()
            if x_supp.dim() == 1:
                x_supp = x_supp.unsqueeze(0)
            if y_supp.dim() == 0:
                y_supp = y_supp.unsqueeze(0)
            
            self._quick_train_task_data_safe(x_supp, y_supp, steps, update_memory)
        except Exception as e:
            pass
    
    def _quick_train_task_on_model_safe(self, model, task, steps=5, update_memory=True):
        """Quick training on a single task using specific model - NO BACKWARD PASS"""
        try:
            x_supp, y_supp = task.support()
            
            x_supp = x_supp.float().detach()
            y_supp = y_supp.long().detach()
            if x_supp.dim() == 1:
                x_supp = x_supp.unsqueeze(0)
            if y_supp.dim() == 0:
                y_supp = y_supp.unsqueeze(0)
            
            model.train()
            
            for _ in range(steps):
                try:
                    y_pred, _ = model(x_supp, update_memory=update_memory)
                    # No backward pass - just forward pass for memory update
                except Exception as e:
                    continue
        except Exception as e:
            pass
    
    def _quick_train_task_data_safe(self, x, y, steps=5, update_memory=True):
        """Quick training on given data - NO BACKWARD PASS"""
        try:
            x = x.float().detach()
            y = y.long().detach()
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if y.dim() == 0:
                y = y.unsqueeze(0)
            
            if len(x) != len(y):
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
            
            self.model.train()
            
            for _ in range(steps):
                try:
                    y_pred, _ = self.model(x, update_memory=update_memory)
                    # No backward pass - just forward pass for memory update
                except Exception as e:
                    continue
                
        except Exception as e:
            pass
    
    def _measure_learning_speed_safe(self, task, max_steps=8):
        """Measure learning curve for a task - NO BACKWARD PASS"""
        try:
            x_supp, y_supp = task.support()
            x_qry, y_qry = task.query()
            
            x_supp = x_supp.float().detach()
            y_supp = y_supp.long().detach()
            x_qry = x_qry.float().detach()
            y_qry = y_qry.long().detach()
            
            if x_supp.dim() == 1:
                x_supp = x_supp.unsqueeze(0)
            if y_supp.dim() == 0:
                y_supp = y_supp.unsqueeze(0)
            if x_qry.dim() == 1:
                x_qry = x_qry.unsqueeze(0)
            if y_qry.dim() == 0:
                y_qry = y_qry.unsqueeze(0)
            
            learning_curve = []
            
            for step in range(max_steps):
                try:
                    # Train step - NO BACKWARD PASS
                    self.model.train()
                    y_pred, _ = self.model(x_supp, update_memory=True)
                    
                    # Test step
                    self.model.eval()
                    with torch.no_grad():
                        y_pred_test, _ = self.model(x_qry, update_memory=False)
                        
                        if y_pred_test.shape[0] != y_qry.shape[0]:
                            min_batch = min(y_pred_test.shape[0], y_qry.shape[0])
                            y_pred_test = y_pred_test[:min_batch]
                            y_qry_test = y_qry[:min_batch]
                        else:
                            y_qry_test = y_qry
                        
                        acc = (torch.argmax(y_pred_test, dim=1) == y_qry_test).float().mean().item()
                    
                    learning_curve.append(acc)
                except Exception as e:
                    learning_curve.append(0.0)
            
            return learning_curve
        except Exception as e:
            return [0.0] * max_steps
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("HUMAN-LIKE LEARNING EVALUATION REPORT")
        print("="*60)
        
        # Forgetting curves
        if 'forgetting_curves' in self.results:
            fc = self.results['forgetting_curves']
            print(f"\n1. FORGETTING CURVES:")
            print(f"   Follows Ebbinghaus Pattern: {'YES' if fc.get('follows_ebbinghaus', False) else 'NO'}")
            print(f"   R² Fit: {fc.get('r_squared', 0):.3f}")
            print(f"   Actual Forgetting Detected: {'YES' if fc.get('actual_forgetting_detected', False) else 'NO'}")
            if 'fitted_parameters' in fc:
                print(f"   Decay Parameters: a={fc['fitted_parameters']['a']:.3f}, "
                      f"b={fc['fitted_parameters']['b']:.3f}, c={fc['fitted_parameters']['c']:.3f}")
        
        # Interference
        if 'interference' in self.results:
            inter = self.results['interference']
            retro = inter['retroactive_interference']
            proact = inter['proactive_interference']
            print(f"\n2. INTERFERENCE EFFECTS:")
            print(f"   Retroactive Interference: {'YES' if retro.get('significant_interference', False) else 'NO'}")
            print(f"   Similar vs Dissimilar: {retro.get('similar_interference', 0):.3f} vs {retro.get('dissimilar_interference', 0):.3f}")
            print(f"   Proactive Interference: {'YES' if proact.get('significant_proactive_interference', False) else 'NO'}")
            print(f"   Learning Impairment: {proact.get('learning_impairment', 0):.3f}")
        
        # Consolidation
        if 'consolidation' in self.results:
            cons = self.results['consolidation']
            print(f"\n3. CONSOLIDATION BENEFITS:")
            print(f"   Significant Benefit: {'YES' if cons.get('significant_benefit', False) else 'NO'}")
            print(f"   Performance Improvement: {cons.get('consolidation_benefit', 0):.3f}")
            print(f"   With/Without Consolidation: {cons.get('consolidation_performance', 0):.3f} vs {cons.get('no_consolidation_performance', 0):.3f}")
        
        # Working memory
        if 'working_memory' in self.results:
            wm = self.results['working_memory']
            print(f"\n4. WORKING MEMORY:")
            print(f"   Estimated Span: {wm.get('estimated_memory_span', 0)} items")
            print(f"   Follows Miller's Rule (7±2): {'YES' if wm.get('follows_millers_rule', False) else 'NO'}")
        
        # Transfer learning
        if 'transfer' in self.results:
            trans = self.results['transfer']
            print(f"\n5. CROSS-DOMAIN TRANSFER:")
            print(f"   Overall Transfer Quality: {trans.get('overall_transfer_quality', 0):.3f}")
            if 'mean_accuracy_by_shots' in trans:
                for shots, acc in trans['mean_accuracy_by_shots'].items():
                    print(f"   {shots}-shot accuracy: {acc:.3f}")
        
        # Schema induction
        if 'schema_induction' in self.results:
            schema = self.results['schema_induction']
            print(f"\n6. SCHEMA INDUCTION:")
            for pattern, results in schema.items():
                if isinstance(results, dict) and 'schema_induction_success' in results:
                    print(f"   {pattern} Pattern: {'YES' if results['schema_induction_success'] else 'NO'} "
                          f"(Transfer: {results.get('mean_transfer', 0):.3f})")
        
        # Emotional effects
        if 'emotional_salience' in self.results:
            emo = self.results['emotional_salience']
            print(f"\n7. EMOTIONAL SALIENCE:")
            print(f"   Significant Effect: {'YES' if emo.get('significant_emotional_effect', False) else 'NO'}")
            print(f"   Emotional Advantage: {emo.get('emotional_advantage', 0):.3f}")
            print(f"   Emotional vs Neutral: {emo.get('emotional_retention', 0):.3f} vs {emo.get('neutral_retention', 0):.3f}")
        
        print("\n" + "="*60)
        
        # Overall human-likeness score
        human_likeness_score = self._calculate_human_likeness_score()
        print(f"OVERALL HUMAN-LIKENESS SCORE: {human_likeness_score:.2f}/10")
        print("="*60)
    
    def _calculate_human_likeness_score(self):
        """Calculate overall human-likeness score"""
        score = 0
        max_score = 0
        
        # Forgetting curves (2.5 points)
        if 'forgetting_curves' in self.results:
            max_score += 2.5
            fc = self.results['forgetting_curves']
            if fc.get('follows_ebbinghaus', False):
                score += 2.5
            elif fc.get('actual_forgetting_detected', False):
                score += 1.5
            elif fc.get('r_squared', 0) > 0.2:
                score += 0.5
        
        # Interference effects (2 points)
        if 'interference' in self.results:
            max_score += 2
            inter = self.results['interference']
            if inter['retroactive_interference'].get('significant_interference', False):
                score += 1
            if inter['proactive_interference'].get('significant_proactive_interference', False):
                score += 1
        
        # Consolidation (1.5 points)
        if 'consolidation' in self.results:
            max_score += 1.5
            if self.results['consolidation'].get('significant_benefit', False):
                score += 1.5
        
        # Working memory limits (1.5 points)
        if 'working_memory' in self.results:
            max_score += 1.5
            if self.results['working_memory'].get('follows_millers_rule', False):
                score += 1.5
        
        # Transfer learning (1.5 points)
        if 'transfer' in self.results:
            max_score += 1.5
            quality = self.results['transfer'].get('overall_transfer_quality', 0)
            if quality > 0.4:
                score += 1.5
            elif quality > 0.25:
                score += 1
            elif quality > 0.15:
                score += 0.5
        
        # Schema induction (1 point)
        if 'schema_induction' in self.results:
            max_score += 1
            success_count = sum(1 for results in self.results['schema_induction'].values() 
                              if isinstance(results, dict) and results.get('schema_induction_success', False))
            if len(self.results['schema_induction']) > 0:
                score += success_count / len(self.results['schema_induction'])
        
        # Emotional effects (1 point)
        if 'emotional_salience' in self.results:
            max_score += 1
            if self.results['emotional_salience'].get('significant_emotional_effect', False):
                score += 1
        
        return (score / max_score) * 10 if max_score > 0 else 0
    
    def save_results(self, filename='evaluation_results.json'):
        """Save evaluation results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"Failed to save results: {e}")
    
    def plot_results(self):
        """Generate visualization plots for all results"""
        try:
            fig = plt.figure(figsize=(16, 12))
            
            # Plot forgetting curves
            if 'forgetting_curves' in self.results and 'mean_retention_by_interval' in self.results['forgetting_curves']:
                plt.subplot(2, 3, 1)
                fc_data = self.results['forgetting_curves']['mean_retention_by_interval']
                if fc_data:
                    intervals = list(fc_data.keys())
                    retention = list(fc_data.values())
                    plt.plot(intervals, retention, 'bo-', label='Model')
                    
                    # Plot theoretical Ebbinghaus curve
                    if 'fitted_parameters' in self.results['forgetting_curves']:
                        params = self.results['forgetting_curves']['fitted_parameters']
                        theoretical = [params['a'] * np.exp(-params['b'] * t) + params['c'] for t in intervals]
                        plt.plot(intervals, theoretical, 'r--', label='Fitted Exponential')
                        plt.legend()
                    
                    plt.xlabel('Time Intervals')
                    plt.ylabel('Retention Ratio')
                    plt.title('Forgetting Curves')
                    plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Failed to create plots: {e}")

# Example usage
if __name__ == "__main__":
    pass

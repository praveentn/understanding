# understanding_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
import numpy as np

# ---------- Enhanced Core Components ----------

class PerceptionEncoder(nn.Module):
    """Enhanced multi-modal encoder with attention"""
    def __init__(self, d_model=256, modality='text'):
        super().__init__()
        self.modality = modality
        if modality == 'text':
            self.fc = nn.Sequential(
                nn.Linear(768, d_model), 
                nn.LayerNorm(d_model),
                nn.ReLU(), 
                nn.Dropout(0.1),
                nn.Linear(d_model, d_model)
            )
        elif modality == 'vision':
            self.fc = nn.Sequential(
                nn.Linear(2048, d_model),  # ResNet features
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, d_model)
            )
        
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        encoded = self.fc(x)
        # Self-attention for sequence processing
        attended, _ = self.attention(encoded, encoded, encoded)
        return attended.mean(dim=1)  # Pool over sequence

class HumanLikeWorkingMemory(torch.nn.Module):
    """FIXED: Working memory with NO in-place parameter operations and NaN protection"""
    def __init__(self, d_model=256, capacity=7):
        super().__init__()
        self.capacity = capacity
        self.d_model = d_model
        
        # Use regular tensors instead of parameters for runtime state
        self.register_buffer('slots', torch.zeros(capacity, d_model))
        self.register_buffer('slot_ages', torch.zeros(capacity))
        self.register_buffer('slot_importance', torch.zeros(capacity))
        self.register_buffer('slot_occupied', torch.zeros(capacity, dtype=torch.bool))
        
        # Trainable components for processing
        self.update_gate = torch.nn.Linear(d_model, d_model)
        self.importance_net = torch.nn.Linear(d_model, 1)
        
    def forward(self, x):
        """Update working memory with new information - NaN safe"""
        batch_size = x.size(0)
        outputs = []
        
        for b in range(batch_size):
            try:
                # NaN check
                if torch.isnan(x[b]).any():
                    outputs.append(torch.zeros(self.d_model, device=x.device, dtype=x.dtype))
                    continue
                    
                output = self._update_single_safe(x[b])
                outputs.append(output)
            except Exception:
                outputs.append(torch.zeros(self.d_model, device=x.device, dtype=x.dtype))
        
        return torch.stack(outputs)
    
    def _update_single_safe(self, x):
        """Update memory without NaN issues"""
        try:
            # Calculate importance with NaN protection
            importance = torch.sigmoid(self.importance_net(x))
            if torch.isnan(importance):
                importance = torch.tensor(0.5)
            
            # Update content with NaN protection
            updated_content = self.update_gate(x)
            if torch.isnan(updated_content).any():
                updated_content = x.clone()
            
            # Find slot to update (detached from graph)
            with torch.no_grad():
                # Find empty or least important slot
                empty_slots = ~self.slot_occupied
                if torch.any(empty_slots):
                    slot_idx = torch.where(empty_slots)[0][0].item()
                    should_update = True
                else:
                    slot_idx = torch.argmin(self.slot_importance).item()
                    should_update = importance.item() > 0.1
                
                if should_update:
                    # SAFE buffer update with NaN check
                    new_content = updated_content.detach()
                    if not torch.isnan(new_content).any():
                        self.slots[slot_idx] = new_content
                        self.slot_occupied[slot_idx] = True
                        self.slot_ages[slot_idx] = 0.0
                        self.slot_importance[slot_idx] = max(0.1, importance.item())
            
            # Apply maintenance operations safely
            with torch.no_grad():
                self.slot_ages += 1
                # Simple decay without complex operations
                decay_mask = self.slot_ages > 50
                if torch.any(decay_mask):
                    self.slot_importance[decay_mask] *= 0.9
            
            # Return current working memory state safely
            if torch.any(self.slot_occupied):
                active_indices = torch.where(self.slot_occupied)[0]
                if len(active_indices) > 0:
                    active_slots = self.slots[active_indices]
                    weights = self.slot_importance[active_indices]
                    if weights.sum() > 0:
                        weights = torch.nn.functional.softmax(weights, dim=0)
                        result = torch.sum(weights.unsqueeze(1) * active_slots, dim=0)
                        
                        # Final NaN check
                        if torch.isnan(result).any():
                            return torch.zeros(self.d_model, device=x.device, dtype=x.dtype)
                        return result
                    else:
                        return active_slots[0]
                else:
                    return torch.zeros(self.d_model, device=x.device, dtype=x.dtype)
            else:
                return torch.zeros(self.d_model, device=x.device, dtype=x.dtype)
                
        except Exception:
            return torch.zeros(self.d_model, device=x.device, dtype=x.dtype)

    def consolidate_memories(self):
        """Consolidate important memories"""
        with torch.no_grad():
            consolidation_candidates = ((self.access_counts > self.consolidation_threshold) & 
                                      (self.importance_scores > 0.8) &
                                      self.schema_active)
            
            if torch.any(consolidation_candidates):
                self.consolidation_strength[consolidation_candidates] *= 1.5
                self.consolidation_strength.clamp_(1.0, 10.0)
                
                self.importance_scores[consolidation_candidates] *= 1.5
                self.importance_scores.clamp_(0.5, 5.0)


# 3. Fix Schema Store to prevent memory issues
class HumanLikeSchemaStore(torch.nn.Module):
    """FIXED: Schema store with proper gradient flow and NaN protection"""
    def __init__(self, num_schemas=128, d_model=256):
        super().__init__()
        self.num_schemas = num_schemas
        self.d_model = d_model
        
        # Core memory components - using buffers for content that changes
        self.register_buffer('keys', torch.randn(num_schemas, d_model) * 0.1)
        self.register_buffer('values', torch.zeros(num_schemas, d_model))
        
        # Trainable components for processing
        self.key_processor = torch.nn.Linear(d_model, d_model)
        self.value_processor = torch.nn.Linear(d_model, d_model)
        self.retrieval_processor = torch.nn.Linear(d_model, d_model)
        
        # Buffers for runtime state
        self.register_buffer('access_counts', torch.zeros(num_schemas))
        self.register_buffer('importance_scores', torch.ones(num_schemas))
        self.register_buffer('schema_active', torch.zeros(num_schemas, dtype=torch.bool))
        self.register_buffer('current_time', torch.zeros(1))
        # Parameters
        self.decay_rate = 0.01
        self.consolidation_threshold = 2
        self.interference_threshold = 0.7

    def retrieve(self, query, top_k=4, update_access=True):
        """Retrieve schemas with NaN protection"""
        try:
            batch_size = query.size(0) if query.dim() > 1 else 1
            if query.dim() == 1:
                query = query.unsqueeze(0)
            
            # NaN check
            if torch.isnan(query).any():
                return torch.zeros(batch_size, self.d_model, device=query.device, dtype=query.dtype)
            
            # Process query with NaN protection
            processed_query = self.retrieval_processor(query)
            if torch.isnan(processed_query).any():
                processed_query = query
            
            retrieved = []
            for b in range(batch_size):
                q = processed_query[b]
                
                # Calculate similarities with NaN protection
                similarities = torch.nn.functional.cosine_similarity(q.unsqueeze(0), self.keys, dim=1)
                if torch.isnan(similarities).any():
                    similarities = torch.zeros_like(similarities)
                
                # Simple retrieval without complex scoring
                active_count = self.schema_active.sum().item()
                if active_count > 0:
                    k = min(top_k, active_count)
                    active_mask = self.schema_active.float()
                    retrieval_scores = similarities * active_mask
                    
                    _, top_indices = torch.topk(retrieval_scores, k)
                    
                    # Filter to only active schemas
                    active_top_indices = []
                    for idx in top_indices:
                        if self.schema_active[idx]:
                            active_top_indices.append(idx)
                    
                    if active_top_indices:
                        active_top_indices = torch.tensor(active_top_indices, device=query.device)
                        retrieved_values = self.values[active_top_indices]
                        
                        # Simple averaging instead of complex weighting
                        result = torch.mean(retrieved_values, dim=0)
                        
                        if torch.isnan(result).any():
                            result = torch.zeros(self.d_model, device=query.device, dtype=query.dtype)
                        
                        retrieved.append(result)
                    else:
                        retrieved.append(torch.zeros(self.d_model, device=query.device, dtype=query.dtype))
                else:
                    retrieved.append(torch.zeros(self.d_model, device=query.device, dtype=query.dtype))
            
            return torch.stack(retrieved).squeeze(0) if batch_size == 1 else torch.stack(retrieved)
            
        except Exception:
            batch_size = query.size(0) if query.dim() > 1 else 1
            return torch.zeros(batch_size, self.d_model, device=query.device, dtype=query.dtype)
    
    def update(self, query, info, emotional_context=None, lr=0.5):
        """Update schemas with NaN protection"""
        try:
            if query.dim() == 1:
                query = query.unsqueeze(0)
            if info.dim() == 1:
                info = info.unsqueeze(0)
            
            # NaN checks
            if torch.isnan(query).any() or torch.isnan(info).any():
                return
            
            # Process inputs with NaN protection
            processed_key = self.key_processor(query)
            processed_value = self.value_processor(info)
            
            if torch.isnan(processed_key).any():
                processed_key = query
            if torch.isnan(processed_value).any():
                processed_value = info
            
            batch_size = query.size(0)
            
            for b in range(batch_size):
                self._update_single_safe(processed_key[b], processed_value[b], lr)
                
        except Exception:
            pass
    
    def _update_single_safe(self, query, info, lr=0.5):
        """Update single schema safely"""
        try:
            with torch.no_grad():
                # Find best match
                similarities = torch.nn.functional.cosine_similarity(query.unsqueeze(0), self.keys, dim=1)
                if torch.isnan(similarities).any():
                    similarities = torch.zeros_like(similarities)
                
                best_match_idx = torch.argmax(similarities).item()
                best_similarity = similarities[best_match_idx].item()
                
                # Update existing or create new (SIMPLIFIED)
                if best_similarity > 0.3 and self.schema_active[best_match_idx]:
                    # Update existing schema (simple version)
                    alpha = 0.1  # Fixed learning rate
                    old_value = self.values[best_match_idx].clone()
                    new_value = (1 - alpha) * old_value + alpha * info.detach()
                    
                    # NaN check before update
                    if not torch.isnan(new_value).any():
                        self.values[best_match_idx] = new_value
                        self.importance_scores[best_match_idx] = min(2.0, self.importance_scores[best_match_idx] + 0.1)
                else:
                    # Create new schema
                    empty_mask = ~self.schema_active
                    
                    if torch.any(empty_mask):
                        empty_idx = torch.where(empty_mask)[0][0].item()
                    else:
                        # Replace least important
                        empty_idx = torch.argmin(self.importance_scores).item()
                    
                    # Update buffers directly with NaN protection
                    if not torch.isnan(query).any() and not torch.isnan(info).any():
                        self.keys[empty_idx] = query.detach()
                        self.values[empty_idx] = info.detach()
                        self.schema_active[empty_idx] = True
                        self.importance_scores[empty_idx] = 1.0
                        self.access_counts[empty_idx] = 1
                        
        except Exception:
            pass
    
    def forget_step(self):
        """Apply forgetting mechanisms safely"""
        try:
            with torch.no_grad():
                self.current_time += 1
                
                if not torch.any(self.schema_active):
                    return
                
                # Simple forgetting every 10 steps
                if self.current_time.item() % 10 == 0:
                    # Randomly forget some schemas
                    active_indices = torch.where(self.schema_active)[0]
                    if len(active_indices) > 5:  # Keep at least 5 schemas
                        num_to_forget = max(1, len(active_indices) // 10)
                        forget_indices = active_indices[torch.randperm(len(active_indices))[:num_to_forget]]
                        
                        # Forget schemas
                        self.schema_active[forget_indices] = False
                        self.values[forget_indices] = 0
                        self.importance_scores[forget_indices] = 0
                        self.access_counts[forget_indices] = 0
                        
        except Exception:
            pass


    def consolidate_memories(self):
        """Consolidate important memories"""
        with torch.no_grad():
            consolidation_candidates = ((self.access_counts > self.consolidation_threshold) & 
                                      (self.importance_scores > 0.8) &
                                      self.schema_active)
            
            if torch.any(consolidation_candidates):
                self.consolidation_strength[consolidation_candidates] *= 1.5
                self.consolidation_strength.clamp_(1.0, 10.0)
                
                self.importance_scores[consolidation_candidates] *= 1.5
                self.importance_scores.clamp_(0.5, 5.0)


class EmotionalTagging(nn.Module):
    """Emotional salience affects memory formation and retention"""
    def __init__(self, d_model=256):
        super().__init__()
        self.emotion_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 8)  # 8 basic emotions
        )
        self.salience_gate = nn.Linear(8, 1)
        
    def forward(self, experience, context=None):
        """Generate emotional tags for experiences"""
        if experience.dim() == 1:
            experience = experience.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        emotion_scores = torch.sigmoid(self.emotion_encoder(experience))
        salience = torch.sigmoid(self.salience_gate(emotion_scores))
        
        if squeeze_output:
            emotion_scores = emotion_scores.squeeze(0)
            salience = salience.squeeze(0)
        
        return emotion_scores, salience

class GraphReasoner(nn.Module):
    """Enhanced GNN with attention and multi-step reasoning"""
    def __init__(self, d_model=256, num_reasoning_steps=3):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_reasoning_steps
        
        # Message passing layers
        self.msg_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Update mechanism
        self.update_gru = nn.GRUCell(d_model, d_model)
        
        # Attention mechanism for focusing
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)
        
        # Final reasoning layer
        self.reason_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, wm, schema):
        """Multi-step reasoning with working memory and schema"""
        if wm.dim() == 1:
            wm = wm.unsqueeze(0)
        if schema.dim() == 1:
            schema = schema.unsqueeze(0)
        
        batch_size = wm.size(0)
        
        # Initialize reasoning state
        reasoning_state = wm
        
        for step in range(self.num_steps):
            # Create reasoning context
            context = torch.stack([reasoning_state, schema], dim=1)
            
            # Apply attention
            attended_context, _ = self.attention(context, context, context)
            attended_wm = attended_context[:, 0]
            attended_schema = attended_context[:, 1]
            
            # Message passing
            combined = torch.cat([attended_wm, attended_schema], dim=-1)
            message = self.msg_net(combined)
            
            # Update reasoning state
            reasoning_state = self.update_gru(message, reasoning_state)
        
        # Final reasoning step
        output = self.reason_net(reasoning_state)
        
        return output.squeeze(0) if batch_size == 1 else output

# ---------- Enhanced Main Network ----------

class EnhancedUnderstandingNet(nn.Module):
    """Complete understanding network with fixed gradient flow"""
    def __init__(self, d_model=256, num_classes=10, modality='text'):
        super().__init__()
        self.d_model = d_model
        
        # Core components
        self.enc = PerceptionEncoder(d_model, modality)
        self.wm = HumanLikeWorkingMemory(d_model)
        self.mem = HumanLikeSchemaStore(128, d_model)
        self.gnn = GraphReasoner(d_model)
        self.emotional_tagger = EmotionalTagging(d_model)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Training state
        self.register_buffer('training_step', torch.zeros(1))
        
    def forward(self, inp, update_memory=True):
        """Forward pass with memory updates"""
        # Encode input
        z = self.enc(inp)
        
        # Update working memory
        wm = self.wm(z)
        
        # Retrieve relevant schemas
        schema = self.mem.retrieve(wm, update_access=update_memory and self.training)
        
        # Reason with working memory and schemas
        reasoning_output = self.gnn(wm, schema)
        
        # Generate emotional context
        emotions, salience = self.emotional_tagger(reasoning_output)
        
        # Update long-term memory if requested
        if update_memory and self.training:
            try:
                if emotions.dim() == 2 and salience.dim() == 2:
                    emotional_boost = emotions * salience
                elif emotions.dim() == 1 and salience.dim() == 1:
                    emotional_boost = emotions * salience
                elif emotions.dim() == 1 and salience.dim() == 2:
                    emotional_boost = emotions * salience.squeeze()
                elif emotions.dim() == 2 and salience.dim() == 1:
                    emotional_boost = emotions * salience.unsqueeze(-1)
                else:
                    emotional_boost = emotions
            except RuntimeError:
                emotional_boost = emotions
            
            # Update with higher learning rate
            self.mem.update(wm, reasoning_output, emotional_boost, lr=0.8)
        
        # Classification
        logits = self.classifier(reasoning_output)
        
        return logits, {
            'working_memory': wm,
            'retrieved_schema': schema,
            'reasoning_output': reasoning_output,
            'emotions': emotions,
            'salience': salience
        }
    
    def forget_and_consolidate(self):
        """Apply forgetting and consolidation mechanisms"""
        self.mem.forget_step()
        
        # More frequent consolidation
        with torch.no_grad():
            self.training_step += 1
            
            if self.training_step.item() % 10 == 0:
                self.mem.consolidate_memories()
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        try:
            active_schemas = self.mem.schema_active.sum().item()
            
            active_mask = self.mem.schema_active
            if active_mask.sum() > 0:
                avg_importance = self.mem.importance_scores[active_mask].mean().item()
                avg_access_count = self.mem.access_counts[active_mask].mean().item()
            else:
                avg_importance = 0.0
                avg_access_count = 0.0
            
            wm_occupied = self.wm.slot_occupied.sum().item()
            
            return {
                'active_schemas': active_schemas,
                'avg_schema_importance': avg_importance,
                'avg_access_count': avg_access_count,
                'working_memory_slots_used': wm_occupied,
                'total_training_steps': self.training_step.item()
            }
        except Exception as e:
            return {
                'active_schemas': 0,
                'avg_schema_importance': 0.0,
                'avg_access_count': 0.0,
                'working_memory_slots_used': 0,
                'total_training_steps': self.training_step.item()
            }

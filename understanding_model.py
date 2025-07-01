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

class HumanLikeWorkingMemory(nn.Module):
    """Working memory with human-like capacity limits and interference"""
    def __init__(self, d_model=256, capacity=7):  # Miller's 7±2 rule
        super().__init__()
        self.capacity = capacity
        self.d_model = d_model
        
        # Memory slots - these are trainable parameters
        self.slots = nn.Parameter(torch.zeros(capacity, d_model))
        
        # Register state variables as buffers (non-trainable runtime state)
        self.register_buffer('slot_ages', torch.zeros(capacity))
        self.register_buffer('slot_importance', torch.zeros(capacity))
        self.register_buffer('slot_occupied', torch.zeros(capacity, dtype=torch.bool))
        
        # Update mechanisms - these are trainable
        self.erase_gate = nn.Linear(d_model, d_model)
        self.add_gate = nn.Linear(d_model, d_model)
        self.importance_net = nn.Linear(d_model, 1)
        
        # Interference parameters
        self.interference_threshold = 0.8
        self.decay_rate = 0.05
        
    def forward(self, x):
        """Update working memory with new information"""
        batch_size = x.size(0)
        outputs = []
        
        for b in range(batch_size):
            output = self._update_single_safe(x[b])
            outputs.append(output)
        
        return torch.stack(outputs)
    
    def _update_single_safe(self, x):
        """Update working memory for single sample with gradient-safe operations"""
        # Calculate importance of new information (this creates gradients)
        importance_logit = self.importance_net(x)
        importance = torch.sigmoid(importance_logit)
        
        # Calculate update content (this creates gradients)
        erase = torch.sigmoid(self.erase_gate(x))
        add = torch.tanh(self.add_gate(x))
        
        # Find slot to update - FIXED LOGIC
        with torch.no_grad():
            # Find first empty slot or least important slot
            empty_slots = ~self.slot_occupied
            if torch.any(empty_slots):
                # Use first empty slot
                slot_idx = torch.where(empty_slots)[0][0].item()
                should_update = True
            else:
                # Replace least important slot
                slot_idx = torch.argmin(self.slot_importance).item()
                should_update = importance.item() > 0.3  # Lower threshold for activation
        
        # Perform the update - FIXED to ensure activation
        if should_update:
            # Create new slot content with gradients
            new_content = add * importance.squeeze()
            
            # Update slot content while preserving gradients
            self.slots.data[slot_idx] = new_content.detach()
            
            # Update state variables without gradients
            with torch.no_grad():
                self.slot_ages[slot_idx] = 0.0
                self.slot_importance[slot_idx] = importance.item()
                self.slot_occupied[slot_idx] = True  # CRITICAL FIX: Mark as occupied
        
        # Apply maintenance operations without gradients
        with torch.no_grad():
            self.slot_ages += 1
            self._apply_interference_and_decay()
        
        # Return current working memory state (maintains gradients through fresh computation)
        if torch.any(self.slot_occupied):
            # Create fresh computation for gradients
            active_indices = torch.where(self.slot_occupied)[0]
            if len(active_indices) > 0:
                # Compute weighted average of active slots
                active_slots = self.slots[active_indices]
                weights = self.slot_importance[active_indices] / (self.slot_importance[active_indices].sum() + 1e-8)
                return torch.sum(weights.unsqueeze(-1) * active_slots, dim=0)
            else:
                return torch.zeros(self.d_model, device=x.device, dtype=x.dtype)
        else:
            return torch.zeros(self.d_model, device=x.device, dtype=x.dtype)
    
    def _apply_interference_and_decay(self):
        """Apply interference between similar memories and temporal decay"""
        if not torch.any(self.slot_occupied):
            return
        
        active_indices = torch.where(self.slot_occupied)[0]
        if len(active_indices) <= 1:
            return
            
        active_slots = self.slots[active_indices]
        
        # Calculate similarities between active slots
        if active_slots.size(0) > 1:
            similarities = F.cosine_similarity(
                active_slots.unsqueeze(1), 
                active_slots.unsqueeze(0), 
                dim=2
            )
            
            # Apply interference for highly similar memories
            interference_mask = (similarities > self.interference_threshold) & (similarities < 1.0)
            
            if torch.any(interference_mask):
                noise_std = 0.05
                noise = torch.randn_like(active_slots) * noise_std
                # Apply noise proportional to interference
                interference_strength = interference_mask.float().sum(dim=1, keepdim=True)
                self.slots.data[active_indices] += noise * interference_strength
        
        # Temporal decay - REDUCED to prevent too aggressive forgetting
        decay_factors = torch.exp(-self.decay_rate * 0.1 * self.slot_ages[active_indices])  # Slower decay
        self.slots.data[active_indices] *= decay_factors.unsqueeze(-1)
        
        # Remove very weak memories - HIGHER threshold to keep more memories
        weak_mask = torch.norm(self.slots[active_indices], dim=1) < 0.05  # Lower threshold
        if torch.any(weak_mask):
            weak_indices = active_indices[weak_mask]
            self.slot_occupied[weak_indices] = False
            self.slots.data[weak_indices] = 0
            self.slot_importance[weak_indices] = 0

class HumanLikeSchemaStore(nn.Module):
    """Enhanced schema store with forgetting curves and consolidation"""
    def __init__(self, num_schemas=128, d_model=256):
        super().__init__()
        self.num_schemas = num_schemas
        self.d_model = d_model
        
        # Core memory components (trainable)
        self.keys = nn.Parameter(torch.randn(num_schemas, d_model) * 0.1)
        self.values = nn.Parameter(torch.zeros(num_schemas, d_model))
        
        # Forgetting mechanisms (non-trainable runtime state)
        self.register_buffer('access_counts', torch.zeros(num_schemas))
        self.register_buffer('last_access', torch.zeros(num_schemas))
        self.register_buffer('importance_scores', torch.ones(num_schemas))
        self.register_buffer('creation_time', torch.zeros(num_schemas))
        self.register_buffer('consolidation_strength', torch.ones(num_schemas))
        
        # Emotional and salience tagging
        self.register_buffer('emotional_tags', torch.zeros(num_schemas, 8))  # 8 basic emotions
        self.register_buffer('schema_active', torch.zeros(num_schemas, dtype=torch.bool))
        
        # Learning parameters
        self.decay_rate = 0.001  # REDUCED for slower forgetting
        self.consolidation_threshold = 5  # LOWERED for easier consolidation
        self.interference_threshold = 0.85
        self.current_time = 0
        
    def retrieve(self, query, top_k=4, update_access=True):
        """Retrieve schemas with forgetting-aware scoring"""
        batch_size = query.size(0) if query.dim() > 1 else 1
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        retrieved = []
        for b in range(batch_size):
            q = query[b]
            
            # Calculate base similarities (maintains gradients)
            similarities = F.cosine_similarity(q.unsqueeze(0), self.keys, dim=1)
            
            # Apply forgetting-aware scoring (no gradients for state)
            with torch.no_grad():
                time_since_access = self.current_time - self.last_access
                forgetting_factor = torch.exp(-self.decay_rate * time_since_access)
                
                # Boost important and emotionally salient memories
                emotional_boost = self.emotional_tags.sum(dim=1) * 0.1
                importance_boost = self.importance_scores * 0.2
                consolidation_boost = torch.log(1 + self.consolidation_strength) * 0.1
                
                # Combine with similarities (gradients preserved for similarities)
                state_boost = emotional_boost + importance_boost + consolidation_boost
                active_mask = self.schema_active.float()
            
            # Final retrieval scores (maintains gradients)
            retrieval_scores = similarities * forgetting_factor.detach() + state_boost.detach()
            retrieval_scores = retrieval_scores * active_mask.detach()
            
            # Get top-k
            active_count = self.schema_active.sum().item()
            if active_count > 0:
                k = min(top_k, active_count)
                top_scores, top_indices = torch.topk(retrieval_scores, k)
                
                if len(top_indices) > 0:
                    retrieved_values = self.values[top_indices]
                    # Weighted average based on retrieval scores (maintains gradients)
                    weights = F.softmax(top_scores, dim=0)
                    retrieved_schema = torch.sum(weights.unsqueeze(1) * retrieved_values, dim=0)
                    
                    # Update access statistics without gradients
                    if update_access:
                        with torch.no_grad():
                            self._update_access_stats(top_indices)
                    
                    retrieved.append(retrieved_schema)
                else:
                    retrieved.append(torch.zeros(self.d_model, device=query.device, dtype=query.dtype))
            else:
                retrieved.append(torch.zeros(self.d_model, device=query.device, dtype=query.dtype))
        
        return torch.stack(retrieved)
    
    def update(self, query, info, emotional_context=None, lr=0.1):
        """Update schemas with new information"""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if info.dim() == 1:
            info = info.unsqueeze(0)
        
        batch_size = query.size(0)
        
        for b in range(batch_size):
            q, inf = query[b], info[b]
            # Handle emotional context properly for each batch item
            if emotional_context is not None:
                if emotional_context.dim() == 2 and b < emotional_context.size(0):
                    emo_ctx = emotional_context[b]
                elif emotional_context.dim() == 1:
                    emo_ctx = emotional_context if b == 0 else None
                else:
                    emo_ctx = None
            else:
                emo_ctx = None
            self._update_single_safe(q, inf, emo_ctx, lr)
    
    def _update_single_safe(self, query, info, emotional_context=None, lr=0.1):
        """Update single schema with gradient-safe operations"""
        # Find most similar existing schema (maintains gradients)
        similarities = F.cosine_similarity(query.unsqueeze(0), self.keys, dim=1)
        
        with torch.no_grad():
            best_match_idx = torch.argmax(similarities).item()
            best_similarity = similarities[best_match_idx].item()
            target_idx = best_match_idx  # Default target for emotional context
        
        # FIXED LOGIC: More aggressive schema creation
        if best_similarity > 0.5 and self.schema_active[best_match_idx]:  # Lowered threshold
            # Update existing schema using .data to avoid in-place operation on parameters
            with torch.no_grad():
                update_strength = lr * self.consolidation_strength[best_match_idx].item()
                # Compute the update without in-place operations on parameters
                current_value = self.values[best_match_idx].clone()
                new_value = current_value + update_strength * (info - current_value)
                self.values.data[best_match_idx] = new_value.detach()
                
                # Update importance
                self.importance_scores[best_match_idx] = min(2.0, 
                    self.importance_scores[best_match_idx] + 0.1)
                
                target_idx = best_match_idx
        else:
            # Create new schema - ALWAYS CREATE if similarity is low
            with torch.no_grad():
                # Find empty slot or replace least important
                empty_mask = ~self.schema_active
                if torch.any(empty_mask):
                    empty_idx = torch.where(empty_mask)[0][0].item()
                else:
                    empty_idx = torch.argmin(self.importance_scores).item()
                
                # CRITICAL FIX: Always create schema if we have room or need to replace
                self.keys.data[empty_idx] = query.detach()
                self.values.data[empty_idx] = info.detach()
                self.schema_active[empty_idx] = True  # MARK AS ACTIVE
                self.creation_time[empty_idx] = self.current_time
                self.importance_scores[empty_idx] = 1.0
                self.consolidation_strength[empty_idx] = 1.0
                self.access_counts[empty_idx] = 1  # Initialize access count
                self.last_access[empty_idx] = self.current_time
                target_idx = empty_idx
        
        # Add emotional context if provided - FIXED for variable sizes
        if emotional_context is not None and isinstance(emotional_context, torch.Tensor):
            with torch.no_grad():
                # Handle different emotional context shapes
                if emotional_context.dim() == 2:
                    # Take first row if batch dimension exists
                    emo_flat = emotional_context[0] if emotional_context.size(0) > 0 else emotional_context.flatten()
                else:
                    emo_flat = emotional_context.flatten()
                
                # Ensure we don't exceed the 8 emotion slots
                num_emotions = min(8, emo_flat.numel())
                if num_emotions > 0:
                    self.emotional_tags[target_idx, :num_emotions] = emo_flat[:num_emotions].detach()
                    # Fill remaining slots with zeros if needed
                    if num_emotions < 8:
                        self.emotional_tags[target_idx, num_emotions:] = 0
    
    def _update_access_stats(self, indices):
        """Update access statistics for retrieved schemas"""
        self.access_counts[indices] += 1
        self.last_access[indices] = self.current_time
        
        # Boost importance for frequently accessed schemas
        frequent_mask = self.access_counts[indices] > 3  # Lowered threshold
        self.importance_scores[indices[frequent_mask]] *= 1.05
        self.importance_scores.clamp_(0, 2.0)
    
    def forget_step(self):
        """Apply forgetting mechanisms"""
        with torch.no_grad():
            self.current_time += 1
            
            # Only apply forgetting if we have active schemas
            if not torch.any(self.schema_active):
                return
            
            # 1. Temporal decay (Ebbinghaus curve) - MUCH slower
            time_since_access = self.current_time - self.last_access
            decay_factor = torch.exp(-self.decay_rate * time_since_access)
            
            # 2. Importance-based retention
            retention_prob = self.importance_scores * decay_factor
            
            # 3. Emotional enhancement (emotional memories persist longer)
            emotional_strength = self.emotional_tags.sum(dim=1)
            retention_prob += emotional_strength * 0.1
            
            # 4. Consolidation protection
            consolidation_protection = torch.log(1 + self.consolidation_strength) * 0.2
            retention_prob += consolidation_protection
            
            # 5. Random forgetting for very old, unused memories - MUCH more conservative
            old_unused_mask = ((time_since_access > 100) &  # Longer threshold
                              (self.access_counts < 1) & 
                              self.schema_active)
            retention_prob[old_unused_mask] *= 0.7  # Less aggressive forgetting
            
            # Apply probabilistic forgetting - MUCH more conservative
            forget_probs = 1 - torch.clamp(retention_prob, 0, 1)
            forget_probs *= 0.1  # Make forgetting 10x less likely
            forget_mask = (torch.rand_like(forget_probs) < forget_probs) & self.schema_active
            
            # Gradual degradation instead of complete removal
            degradation_factor = 0.95  # Less aggressive degradation
            self.values.data[forget_mask] *= degradation_factor
            self.importance_scores[forget_mask] *= degradation_factor
            
            # Remove completely degraded schemas - MUCH higher threshold
            very_weak_mask = (torch.norm(self.values, dim=1) < 0.01) & self.schema_active
            if torch.any(very_weak_mask):
                self.schema_active[very_weak_mask] = False
                self.values.data[very_weak_mask] = 0
                self.keys.data[very_weak_mask] = torch.randn_like(self.keys[very_weak_mask]) * 0.1
                self.importance_scores[very_weak_mask] = 0
                self.access_counts[very_weak_mask] = 0
    
    def consolidate_memories(self):
        """Strengthen important memories during consolidation"""
        with torch.no_grad():
            # Identify memories for consolidation - LOWERED thresholds
            consolidation_candidates = ((self.access_counts > self.consolidation_threshold) & 
                                       (self.importance_scores > 0.8) &  # Lowered threshold
                                       self.schema_active)
            
            if torch.any(consolidation_candidates):
                # Strengthen these memories
                self.consolidation_strength[consolidation_candidates] *= 1.2
                self.consolidation_strength.clamp_(1.0, 5.0)
                
                # Make them more resistant to forgetting
                self.importance_scores[consolidation_candidates] *= 1.1
                self.importance_scores.clamp_(0, 2.0)

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
        # Ensure proper batch dimension
        if experience.dim() == 1:
            experience = experience.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        emotion_scores = torch.sigmoid(self.emotion_encoder(experience))  # [batch, 8]
        salience = torch.sigmoid(self.salience_gate(emotion_scores))      # [batch, 1]
        
        # Return consistent shapes
        if squeeze_output:
            emotion_scores = emotion_scores.squeeze(0)  # [8]
            salience = salience.squeeze(0)              # [1]
        
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
            context = torch.stack([reasoning_state, schema], dim=1)  # [batch, 2, d_model]
            
            # Apply attention
            attended_context, _ = self.attention(context, context, context)
            attended_wm = attended_context[:, 0]  # Working memory component
            attended_schema = attended_context[:, 1]  # Schema component
            
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
    """Complete understanding network with human-like learning and forgetting"""
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
        self.training_step = 0
        
    def forward(self, inp, update_memory=True):
        """Forward pass with optional memory updates"""
        # Encode input
        z = self.enc(inp)
        
        # Update working memory - ALWAYS update to ensure activation
        wm = self.wm(z)
        
        # Retrieve relevant schemas
        schema = self.mem.retrieve(wm, update_access=update_memory)
        
        # Reason with working memory and schemas
        reasoning_output = self.gnn(wm, schema)
        
        # Generate emotional context
        emotions, salience = self.emotional_tagger(reasoning_output)
        
        # Update long-term memory if requested - ALWAYS update during training
        if update_memory:
            # FIXED: Robust emotional boost calculation
            try:
                if emotions.dim() == 2 and salience.dim() == 2:
                    # Both have batch dimension: [batch,8] * [batch,1] = [batch,8]
                    emotional_boost = emotions * salience
                elif emotions.dim() == 1 and salience.dim() == 1:
                    # Both are 1D: [8] * [1] = [8]
                    emotional_boost = emotions * salience
                elif emotions.dim() == 1 and salience.dim() == 2:
                    # emotions=[8], salience=[1,1] -> squeeze salience
                    emotional_boost = emotions * salience.squeeze()
                elif emotions.dim() == 2 and salience.dim() == 1:
                    # emotions=[1,8], salience=[1] -> unsqueeze salience
                    emotional_boost = emotions * salience.unsqueeze(-1)
                else:
                    # Fallback: just use emotions without boost
                    emotional_boost = emotions
            except RuntimeError as e:
                # If any broadcasting fails, just use emotions
                print(f"Warning: Emotional boost calculation failed ({e}), using emotions only")
                emotional_boost = emotions
            
            self.mem.update(wm, reasoning_output, emotional_boost)
        
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
        
        # Periodic consolidation (every 50 steps) - More frequent
        if self.training_step % 50 == 0:
            self.mem.consolidate_memories()
        
        self.training_step += 1
    
    def get_memory_stats(self):
        """Get current memory statistics for analysis"""
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
                'total_training_steps': self.training_step
            }
        except Exception as e:
            # Return safe defaults if stats computation fails
            return {
                'active_schemas': 0,
                'avg_schema_importance': 0.0,
                'avg_access_count': 0.0,
                'working_memory_slots_used': 0,
                'total_training_steps': self.training_step
            }

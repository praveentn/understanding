# Holistic Understanding Framework: Human-Like Learning with Forgetting

# 🧠 Enhanced Understanding Framework - Complete Guide

## What This Framework Does

This is a **human-like artificial intelligence system** that learns and forgets like humans do. Unlike traditional AI that never forgets, this system:

### 🎯 **Core Purpose**
- **Mimics human cognition** with realistic memory limitations
- **Learns new tasks quickly** (few-shot learning) like humans
- **Forgets old information** following psychological patterns (Ebbinghaus curve)
- **Consolidates important memories** while discarding irrelevant ones
- **Shows interference effects** between similar memories

---

## 🏗️ System Architecture

### **1. Perception Encoder** (`PerceptionEncoder`)
- **What it does**: Processes raw input (text/images) into internal representations
- **Human analogy**: Your sensory processing - converting what you see/hear into thoughts

### **2. Working Memory** (`HumanLikeWorkingMemory`)
- **What it does**: Temporarily holds 7±2 items (Miller's rule) for immediate processing
- **Human analogy**: Your short-term memory when doing mental math or remembering a phone number
- **Key features**:
  - Limited capacity (7 slots max)
  - Items decay over time
  - Similar items interfere with each other

### **3. Schema Store** (`HumanLikeSchemaStore`) 
- **What it does**: Long-term memory that stores learned patterns and knowledge
- **Human analogy**: Your long-term memory with concepts, facts, and experiences
- **Key features**:
  - Follows forgetting curves (recent memories stronger)
  - Important/emotional memories persist longer
  - Consolidation strengthens frequently used knowledge

### **4. Graph Reasoner** (`GraphReasoner`)
- **What it does**: Combines working memory + schemas to solve problems
- **Human analogy**: Your reasoning process when connecting ideas to make decisions

### **5. Emotional Tagger** (`EmotionalTagging`)
- **What it does**: Marks emotionally significant experiences for better retention
- **Human analogy**: Why you remember emotional events more vividly

---

## 📊 What Makes It "Human-Like"

### **Forgetting Curves (Ebbinghaus)**
```
Memory Retention
     100% |●
          |  ●
          |    ●
          |      ●
          |        ●____
       0% |             ●___________
          +---------------------------→ Time
          0   1hr  1day  1week  1month
```

### **Interference Effects**
- **Proactive**: Old memories interfere with learning new similar info
- **Retroactive**: New learning interferes with old similar memories
- Just like humans struggle with similar phone numbers or passwords!

### **Memory Consolidation**
- Important memories get strengthened during "sleep" (consolidation phases)
- Frequently accessed information becomes more permanent

---

## 🚀 How to Use the Framework

### **Option 1: Simple Launcher (Recommended)**
```bash
python launcher.py
```
Then choose:
- **1** = Quick Demo (5 minutes)
- **2** = Full Training (30-60 minutes) 
- **3** = Custom settings
- **4** = Evaluate existing model

### **Option 2: Command Line**
```bash
# Quick test of basic functionality
python main.py --mode quick_demo

# Full training with comprehensive evaluation
python main.py --mode full_training

# Custom training with your parameters
python main.py --mode custom --iters 100 --d_model 128
```

### **Option 3: Evaluate Pre-trained Model**
```bash
python main.py --mode evaluation_only --model_path results_20250701_191640/model_checkpoint.pth
```

---

## 🔍 How to Verify What the Model Learned

### **1. Check Training Results**
After training, look in the results folder (e.g., `results_20250701_191640/`):

```
results_20250701_191640/
├── demo_results.json          # Raw training data
├── memory_evolution.png       # Memory usage over time
├── comprehensive_analysis.png # Human-likeness analysis
└── training_log.txt          # Detailed logs
```

### **2. Key Metrics to Check**

#### **Memory Dynamics**
```json
"forgetting_events": [
  {
    "iteration": 10,
    "schemas_before": 15,
    "schemas_after": 12,
    "schemas_forgotten": 3
  }
]
```
- **Good**: Schemas are being forgotten over time
- **Bad**: No forgetting (schemas_forgotten = 0 always)

#### **Learning Progress**
```json
"test_accuracies": [
  {"iteration": 10, "accuracy": 0.28},
  {"iteration": 20, "accuracy": 0.36},
  {"iteration": 30, "accuracy": 0.32}
]
```
- **Good**: Accuracy generally improves, with some fluctuation
- **Bad**: No improvement or constant decline

### **3. Human-Likeness Verification**

The full evaluation tests 7 key properties:

#### **✅ Forgetting Curves**
```python
# Check if model follows Ebbinghaus curve
"follows_ebbinghaus": true,
"r_squared": 0.85  # How well it fits (>0.7 is good)
```

#### **✅ Interference Effects**
```python
"retroactive_interference": {
  "significant_interference": true,
  "similar_interference": 0.15,      # Higher = more realistic
  "dissimilar_interference": 0.03    # Should be lower
}
```

#### **✅ Working Memory Limits**
```python
"working_memory": {
  "estimated_memory_span": 7,       # Should be 5-9 (Miller's rule)
  "follows_millers_rule": true
}
```

### **4. Visual Analysis**

#### **Memory Evolution Plot**
Shows how the model's memory changes during training:
- **Active schemas** should fluctuate (learning + forgetting)
- **Working memory usage** should vary
- **Importance scores** should increase for consolidated memories

#### **Forgetting Curves Plot**
Shows retention over time:
- Should follow exponential decay
- Recent memories should be stronger
- Curve should match human psychology research

---

## 🧪 Hands-On Verification Examples

### **Test 1: Quick Memory Check**
```python
# After training, test if model remembers recent vs old tasks
trained_model.eval()

# Test recent memory (should be good)
recent_task = task_generator.sample()
accuracy_recent = test_task_recall(trained_model, recent_task)

# Apply forgetting
for _ in range(50):
    trained_model.mem.forget_step()

# Test same task after forgetting (should be worse)
accuracy_after_forgetting = test_task_recall(trained_model, recent_task)

print(f"Before forgetting: {accuracy_recent:.3f}")
print(f"After forgetting: {accuracy_after_forgetting:.3f}")
print(f"Forgot: {accuracy_recent - accuracy_after_forgetting:.3f}")
```

### **Test 2: Working Memory Capacity**
```python
# Test if model respects 7±2 rule
memory_stats = trained_model.get_memory_stats()
wm_usage = memory_stats['working_memory_slots_used']

print(f"Working memory slots used: {wm_usage}/7")
if 3 <= wm_usage <= 9:
    print("✅ Realistic working memory usage")
else:
    print("❌ Unrealistic working memory usage")
```

### **Test 3: Schema Consolidation**
```python
# Check if important memories get strengthened
stats = trained_model.mem.importance_scores[trained_model.mem.schema_active]
consolidation = trained_model.mem.consolidation_strength[trained_model.mem.schema_active]

print(f"Average importance: {stats.mean():.3f}")
print(f"Average consolidation: {consolidation.mean():.3f}")

# Good signs: Some memories much more important than others
# Indicates selective strengthening
```

---

## 📈 Interpreting Results

### **Good Training Indicators:**
- ✅ Meta-loss decreases over time
- ✅ Test accuracy improves (even if slowly)
- ✅ Memory forgetting events occur regularly
- ✅ Some schemas get consolidated (importance > 1.5)
- ✅ Working memory usage varies (not always 0 or 7)

### **Warning Signs:**
- ❌ No forgetting events (schemas_forgotten always 0)
- ❌ No consolidation (all importance scores = 1.0)
- ❌ Working memory always empty or always full
- ❌ Test accuracy never improves
- ❌ Meta-loss doesn't decrease at all

### **Human-Likeness Score:**
The framework calculates an overall score (0-10):
- **8-10**: Highly human-like across all dimensions
- **6-8**: Good human-like properties with some gaps  
- **4-6**: Some human-like behaviors but inconsistent
- **0-4**: Not very human-like, more like traditional AI

---

## 🎯 Practical Applications

### **Research Applications:**
- **Cognitive Psychology**: Study human memory models
- **Continual Learning**: AI that doesn't catastrophically forget
- **Educational AI**: Systems that learn like students do
- **Adaptive Systems**: AI that adjusts to changing environments

### **Real-World Use Cases:**
- **Personal Assistants**: Remember important things, forget trivial details
- **Educational Software**: Adapt to student learning patterns
- **Recommendation Systems**: Evolve preferences over time
- **Game AI**: Opponents that learn and forget like humans

---

## 🔧 Customization Options

### **Memory Parameters:**
```python
# Adjust forgetting rate
model.mem.decay_rate = 0.02  # Faster forgetting

# Change working memory capacity  
model.wm.capacity = 5  # Smaller capacity

# Modify consolidation threshold
model.mem.consolidation_threshold = 5  # Easier consolidation
```

### **Training Parameters:**
```python
# More/fewer iterations
python main.py --mode custom --iters 200

# Different model sizes
python main.py --mode custom --d_model 512

# Adjusted learning rates
python main.py --mode custom --inner_lr 0.02 --outer_lr 0.001
```

---

## 🚨 Troubleshooting

### **Common Issues:**

1. **No forgetting happening:**
   - Increase `forgetting_freq` parameter
   - Check if tasks are too easy (model not forming memories)

2. **Model not learning:**
   - Reduce learning rates
   - Increase number of training iterations
   - Check task difficulty

3. **Memory always empty:**
   - Reduce `importance_threshold` for memory updates
   - Check if emotional tagging is working

4. **Unrealistic memory patterns:**
   - Adjust `decay_rate` and `interference_threshold`
   - Verify task generator is creating diverse tasks

---

## 🎓 Advanced Usage

### **Creating Custom Tasks:**
```python
class MyTaskGenerator(TaskGenerator):
    def create_my_task(self):
        # Define your specific learning task
        support_x = torch.randn(5, 768)
        support_y = torch.randint(0, 3, (5,))
        query_x = torch.randn(10, 768) 
        query_y = torch.randint(0, 3, (10,))
        
        return Task(support_x, support_y, query_x, query_y,
                   task_info={'type': 'my_custom_task'})
```

### **Custom Evaluation Metrics:**
```python
def my_evaluation(model, tasks):
    # Test specific aspects you care about
    retention_scores = []
    for task in tasks:
        # Your evaluation logic here
        pass
    return {'my_metric': np.mean(retention_scores)}
```

---

This framework represents a significant step toward AI systems that learn and remember like humans do, with realistic cognitive constraints and psychological patterns. The key insight is that **forgetting is not a bug, it's a feature** that enables flexible, adaptive intelligence.


## 1. Extended Architecture Components

### Core Existing Components (Enhanced)
- **PerceptionEncoder**: Multi-modal input processing
- **GatedMemory**: Working memory with attention mechanisms
- **SchemaStore**: Long-term memory with forgetting curves
- **GraphReasoner**: Structured reasoning and inference
- **MetaLearner**: Few-shot adaptation capabilities

### New Human-Inspired Components

#### A. Forgetting Mechanisms
```python
class AdaptiveForgetting(nn.Module):
    """Implements multiple forgetting mechanisms like humans"""
    
    def __init__(self, d_model=256):
        super().__init__()
        # Decay-based forgetting (Ebbinghaus curve)
        self.decay_scheduler = ExponentialDecay()
        
        # Interference-based forgetting
        self.interference_gate = nn.Linear(d_model, 1)
        
        # Importance-based selective forgetting
        self.importance_scorer = nn.Linear(d_model, 1)
        
        # Consolidation mechanism
        self.consolidation_net = nn.GRU(d_model, d_model)
```

#### B. Memory Consolidation System
```python
class MemoryConsolidation(nn.Module):
    """Mimics sleep-like memory consolidation"""
    
    def __init__(self, d_model=256):
        super().__init__()
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.consolidation_transformer = nn.Transformer(d_model, nhead=8, num_layers=2)
        
    def consolidate(self, working_memory, schemas):
        # Replay important experiences
        # Transfer from working memory to long-term storage
        # Strengthen important connections
        pass
```

#### C. Emotional Tagging System
```python
class EmotionalTagging(nn.Module):
    """Emotional salience affects memory formation and retention"""
    
    def __init__(self, d_model=256):
        super().__init__()
        self.emotion_encoder = nn.Linear(d_model, 8)  # 8 basic emotions
        self.salience_gate = nn.Linear(8, 1)
        
    def tag_memory(self, experience, context):
        # Higher emotional salience = stronger memory encoding
        # Affects both storage and forgetting rates
        pass
```

#### D. Curiosity and Exploration Module
```python
class CuriosityDrivenLearning(nn.Module):
    """Drives exploration and learning like human curiosity"""
    
    def __init__(self, d_model=256):
        super().__init__()
        self.prediction_net = nn.Linear(d_model, d_model)
        self.uncertainty_estimator = nn.Linear(d_model, 1)
        
    def intrinsic_motivation(self, state, prediction):
        # High prediction error = high curiosity
        # Guides attention and memory formation
        pass
```

## 2. Enhanced Schema Store with Forgetting

```python
class HumanLikeSchemaStore(nn.Module):
    """Enhanced schema store with human-like forgetting patterns"""
    
    def __init__(self, num_schemas=128, d_model=256):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(num_schemas, d_model))
        self.values = nn.Parameter(torch.zeros(num_schemas, d_model))
        
        # Forgetting mechanisms
        self.access_counts = nn.Parameter(torch.zeros(num_schemas), requires_grad=False)
        self.last_access = nn.Parameter(torch.zeros(num_schemas), requires_grad=False)
        self.importance_scores = nn.Parameter(torch.ones(num_schemas), requires_grad=False)
        self.emotional_tags = nn.Parameter(torch.zeros(num_schemas, 8), requires_grad=False)
        
        # Forgetting parameters
        self.decay_rate = 0.01
        self.interference_threshold = 0.8
        
    def forget_step(self, current_time):
        """Apply forgetting mechanisms"""
        # 1. Temporal decay (Ebbinghaus curve)
        time_since_access = current_time - self.last_access
        decay_factor = torch.exp(-self.decay_rate * time_since_access)
        
        # 2. Importance-based retention
        retention_prob = self.importance_scores * decay_factor
        
        # 3. Emotional enhancement
        emotional_boost = torch.sum(self.emotional_tags, dim=1) * 0.1
        retention_prob += emotional_boost
        
        # 4. Random forgetting for unused memories
        unused_mask = self.access_counts < 2
        retention_prob[unused_mask] *= 0.5
        
        # Apply forgetting
        forget_mask = torch.rand_like(retention_prob) > retention_prob
        self.values.data[forget_mask] *= 0.9  # Gradual degradation
        
    def consolidate_memories(self):
        """Strengthen frequently accessed, important memories"""
        frequent_mask = self.access_counts > torch.quantile(self.access_counts, 0.8)
        self.importance_scores[frequent_mask] *= 1.1
        self.importance_scores.clamp_(0, 2.0)
```

## 3. Working Memory with Human-Like Limitations

```python
class HumanLikeWorkingMemory(nn.Module):
    """Working memory with capacity limits and interference"""
    
    def __init__(self, d_model=256, capacity=7):  # Miller's magic number ±2
        super().__init__()
        self.capacity = capacity
        self.slots = nn.Parameter(torch.zeros(capacity, d_model))
        self.slot_ages = nn.Parameter(torch.zeros(capacity), requires_grad=False)
        self.slot_importance = nn.Parameter(torch.zeros(capacity), requires_grad=False)
        
        self.update_gate = nn.Linear(d_model * 2, d_model)
        self.importance_net = nn.Linear(d_model, 1)
        
    def update(self, new_info):
        """Update working memory with capacity constraints"""
        # Calculate importance of new information
        new_importance = torch.sigmoid(self.importance_net(new_info))
        
        # Find least important slot if at capacity
        if torch.sum(self.slot_ages > 0) >= self.capacity:
            # Replace least important or oldest slot
            replace_scores = self.slot_importance - 0.1 * self.slot_ages
            replace_idx = torch.argmin(replace_scores)
        else:
            # Find empty slot
            replace_idx = torch.argmax((self.slot_ages == 0).float())
        
        # Update slot
        if new_importance > self.slot_importance[replace_idx] or self.slot_ages[replace_idx] == 0:
            self.slots[replace_idx] = new_info
            self.slot_ages[replace_idx] = 1
            self.slot_importance[replace_idx] = new_importance
        
        # Age all slots
        self.slot_ages += 1
        
        # Apply interference and decay
        self._apply_interference()
        
    def _apply_interference(self):
        """Similar memories interfere with each other"""
        similarities = F.cosine_similarity(
            self.slots.unsqueeze(1), 
            self.slots.unsqueeze(0), 
            dim=2
        )
        
        # High similarity causes interference
        interference_mask = (similarities > 0.8) & (similarities < 1.0)
        interference_strength = 0.05
        
        for i in range(self.capacity):
            interfering_slots = interference_mask[i]
            if torch.any(interfering_slots):
                noise = torch.randn_like(self.slots[i]) * interference_strength
                self.slots[i] += noise
```

## 4. Integrated Training Loop with Forgetting

```python
def human_like_meta_train(model, tasks, inner_steps=5, inner_lr=1e-2, 
                         outer_lr=1e-3, iters=1000, forgetting_freq=10):
    """Meta-training with periodic forgetting and consolidation"""
    
    opt = torch.optim.Adam(model.parameters(), lr=outer_lr)
    
    for i in range(iters):
        task = tasks.sample()
        
        # Standard meta-learning inner loop
        fast_weights = {n: p.clone() for n, p in model.named_parameters()}
        
        for _ in range(inner_steps):
            x_supp, y_supp = task.support()
            y_pred = model(x_supp)
            loss = F.cross_entropy(y_pred, y_supp)
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            for (name, _), g in zip(fast_weights.items(), grads):
                fast_weights[name] = fast_weights[name] - inner_lr * g
        
        # Meta-update
        x_qry, y_qry = task.query()
        y_pred = model.forward(x_qry)
        meta_loss = F.cross_entropy(y_pred, y_qry)
        
        opt.zero_grad()
        meta_loss.backward()
        opt.step()
        
        # Periodic forgetting and consolidation
        if i % forgetting_freq == 0:
            with torch.no_grad():
                # Apply forgetting mechanisms
                model.mem.forget_step(current_time=i)
                model.wm.apply_interference()
                
                # Consolidation during "sleep"
                if i % (forgetting_freq * 10) == 0:
                    model.mem.consolidate_memories()
                    model.consolidation.consolidate(model.wm.slots, model.mem.values)
```

## 5. Evaluation Protocols for Human-Like Learning

### A. Forgetting Curve Validation
```python
def test_forgetting_curves(model, retention_intervals=[1, 7, 30, 90]):
    """Test if model follows human-like forgetting patterns"""
    # Train on initial tasks
    # Test retention at different intervals
    # Compare to Ebbinghaus forgetting curve
    pass
```

### B. Interference Effects
```python
def test_interference_effects(model):
    """Test proactive and retroactive interference"""
    # Learn task A
    # Learn similar task B (interference condition)
    # Test retention of task A
    # Compare to control condition
    pass
```

### C. Consolidation Benefits
```python
def test_consolidation_benefits(model):
    """Test if consolidation improves retention"""
    # Learn multiple tasks
    # Apply consolidation to subset
    # Test long-term retention differences
    pass
```

## 6. Implementation Phases

### Phase 1: Core Forgetting Mechanisms (Weeks 1-2)
- Implement temporal decay in SchemaStore
- Add capacity limits to WorkingMemory
- Basic interference mechanisms

### Phase 2: Advanced Memory Dynamics (Weeks 3-4)
- Emotional tagging system
- Importance-based selective forgetting
- Memory consolidation during training

### Phase 3: Curiosity and Exploration (Weeks 5-6)
- Intrinsic motivation system
- Curiosity-driven memory formation
- Adaptive exploration strategies

### Phase 4: Integration and Evaluation (Weeks 7-8)
- Integrate all components
- Comprehensive evaluation suite
- Comparison with baseline models

## 7. Expected Benefits

1. **More Human-Like Learning**: Gradual forgetting of irrelevant information
2. **Better Generalization**: Preventing overfitting through adaptive forgetting
3. **Efficient Memory Usage**: Automatic pruning of unimportant memories
4. **Robust Performance**: Interference resistance and consolidation benefits
5. **Interpretable Behavior**: Memory dynamics that mirror human cognition

## 8. Technical Considerations

- **Memory Efficiency**: Implement sparse updates for large schema stores
- **Stability**: Ensure forgetting doesn't destroy critical knowledge
- **Hyperparameter Sensitivity**: Careful tuning of forgetting rates
- **Evaluation Metrics**: Develop human-cognition-inspired benchmarks

This framework creates a more complete model of human-like learning, incorporating both the strengths (rapid learning, generalization) and limitations (forgetting, interference) that make human cognition robust and adaptive.

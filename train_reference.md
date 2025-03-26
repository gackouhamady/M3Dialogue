# Training Framework for Dialogue Emotion Recognition

## Core Components and Their References

### 1. Data Loading and Sampling
#### `get_IEMOCAP_loaders()`
- **Paper**: [IEMOCAP Dataset](https://sail.usc.edu/iemocap/)  
  - Original multimodal emotion dataset
- **Reference**: [DialogueGCN Implementation](https://github.com/declare-lab/conv-emotion/blob/master/main.py)
  - Data loading pipeline

#### `SubsetRandomSampler`
- **Source**: [PyTorch Documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.SubsetRandomSampler)
  - For train/validation splits

### 2. Training Utilities
#### `seed_everything()`
- **Best Practice**: [Reproducibility in PyTorch](https://pytorch.org/docs/stable/notes/randomness.html)
  - Seed all random number generators

#### `train_or_eval_model()`
- **Paper**: [DialogueRNN (Majumder et al., ACL 2019)](https://arxiv.org/abs/1811.00405)
  - Training protocol (Section 4)
- **Metrics**:
  - Weighted F1: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
  - Masked Accuracy: [DialogueGCN](https://arxiv.org/abs/1908.11540) (Eq. 7)

#### `train_or_eval_graph_model()`
- **Paper**: [DialogueGCN (Ghosal et al., EMNLP 2019)](https://arxiv.org/abs/1908.11540)
  - Graph-specific training (Section 3.4)
  - Edge feature handling (Eq. 1-3)

### 3. Model Components
#### `MaskedNLLLoss`
- **Reference**: 
  - [PyTorch NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)
  - [DialogueRNN Implementation](https://github.com/declare-lab/conv-emotion/blob/master/model.py#L11)

#### Optimizer (Adam)
- **Paper**: [Adam: A Method for Stochastic Optimization (Kingma & Ba, ICLR 2015)](https://arxiv.org/abs/1412.6980)
  - Default hyperparameters (β1=0.9, β2=0.999)

### 4. Evaluation Metrics
| Metric | Implementation | Paper Reference |
|--------|----------------|-----------------|
| Weighted F1 | `sklearn.metrics.f1_score` | [DialogueGCN Evaluation](https://arxiv.org/abs/1908.11540) (Section 5.3) |
| Accuracy | `sklearn.metrics.accuracy_score` | Standard classification metric |
| Confusion Matrix | `sklearn.metrics.confusion_matrix` | Model analysis tool |

### 5. Core Training Loop
```python
for e in range(n_epochs):
    # 1. Forward pass
    # 2. Loss computation 
    # 3. Backward pass
    # 4. Metric calculation
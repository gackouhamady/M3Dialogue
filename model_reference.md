# Mathematical Foundations and Official References

## 1. Core Loss Functions
### MaskedNLLLoss
- **Paper**: [Attention Is All You Need (Vaswani et al., NeurIPS 2017)](https://arxiv.org/abs/1706.03762)
  - Masked attention (Section 3.2.1)

### MaskedMSELoss
- **Paper**: [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Sequence modeling with MSE

### UnMaskedWeightedNLLLoss 
- **Paper**: [Class-Balanced Loss (Cui et al., CVPR 2019)](https://arxiv.org/abs/1901.05555)
  - Eq. 5 for weighted NLL

## 2. Attention Mechanisms
### SimpleAttention
- **Paper**: [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., ICLR 2015)](https://arxiv.org/abs/1409.0473)
  - Basic attention mechanism (Eq. 6-8)

### MatchingAttention
- **Paper**: [Matching Networks for One Shot Learning (Vinyals et al., NeurIPS 2016)](https://arxiv.org/abs/1606.04080)
  - Attention variants (Section 2.1)

### Multi-Head Attention
- **Paper**: [Attention Is All You Need (Vaswani et al., NeurIPS 2017)](https://arxiv.org/abs/1706.03762)
  - Scaled dot-product attention (Eq. 1)

## 3. DialogueRNN Components
### DialogueRNNCell
- **Paper**: [DialogueRNN: An Attentive RNN for Emotion Detection (Majumder et al., ACL 2019)](https://arxiv.org/abs/1811.00405)
  - Speaker-state GRUs (Eq. 1-4)

### Bidirectional DialogueRNN
- **Paper**: [ICON: Interactive Conversational Memory Network (Hazarika et al., EMNLP 2018)](https://arxiv.org/abs/1809.07258)
  - Bidirectional modeling (Section 3.2)

## 4. Graph Components
### MaskedEdgeAttention
- **Paper**: [DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition (Ghosal et al., EMNLP 2019)](https://arxiv.org/abs/1908.11540)
  - Edge weight computation (Eq. 1)

### RGCNConv
- **Paper**: [Modeling Relational Data with Graph Convolutional Networks (Schlichtkrull et al., ESWC 2018)](https://arxiv.org/abs/1703.06103)
  - Relation-specific convolutions (Eq. 2-3)

## 5. CNN Feature Extraction
### CNNFeatureExtractor
- **Paper**: [Convolutional Neural Networks for Sentence Classification (Kim, EMNLP 2014)](https://arxiv.org/abs/1408.5882)
  - Multi-window CNNs (Section 2)

## Implementation Reference
| Component           | Code Reference          | Paper Reference |
|---------------------|-------------------------|-----------------|
| DialogueGCN         | `DialogueGCNModel`      | [EMNLP 2019](https://arxiv.org/abs/1908.11540) |
| Batch Graphify      | `batch_graphify()`      | Algorithm 1 in DialogueGCN |
| Nodal Attention     | `classify_node_features()` | [Graph Attention Networks (ICLR 2018)](https://arxiv.org/abs/1710.10903) |
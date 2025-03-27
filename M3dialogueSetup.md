# Environment Setup for M3Dialogue

## 1. System Requirements
- **GPU**: NVIDIA CUDA ≥ 11.3 (recommended)
- **RAM**: ≥ 16GB (32GB for large datasets)
- **Disk Space**: ≥ 50GB

## 2. Creating the Virtual Environment
```bash
# For Linux/Mac
python -m venv m3dialogue_env
source m3dialogue_env/bin/activate

# For Windows
python -m venv m3dialogue_env
.\m3dialogue_env\Scripts\activate
```
**Note**: Keep the terminal open after activation.

## 3. Installing Core Libraries
```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Essential libraries
pip install transformers numpy pandas tqdm
```

## 4. Multimodal Processing

### Text
```bash
pip install spacy nltk sentence-transformers
python -m spacy download en_core_web_sm  # English model for spaCy
```

### Audio
```bash
pip install librosa torchaudio pydub
```

### Visual
```bash
pip install opencv-python face-alignment mediapipe
```

## 5. Graph Neural Networks
```bash
# DGL (Deep Graph Library)
pip install dgl-cu113 -f https://data.dgl.ai/wheels/cu113/repo.html
## For CPU 
pip install dgl dglgo

# PyTorch Geometric
pip install torch-geometric
# CUDA 11.3 version (GPU)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric
```

## 6. Evaluation and Monitoring
```bash
pip install scikit-learn wandb matplotlib seaborn
```

## 7. Installation Verification
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
```

## 8. requirements.txt File (Optional)
```bash
# Generate the file
pip freeze > requirements.txt

# Reinstall later
pip install -r requirements.txt
```

## Best Practices
### Test Critical Imports:
```python
from transformers import BertModel  # Test BERT
import dgl  # Test GNN
import librosa  # Test audio
```

### W&B Configuration:
```bash
wandb login  # Experiment tracking
```

### Resolving Conflicts:
```bash
pip check  # Check for incompatibilities
```
**Warning**: For CUDA-related issues, check the version with `nvcc --version`

---

## File Features:
1. **Clear sections** with visual hierarchy
2. **Copy-paste ready commands**
3. **Contextual comments** for each step
4. **Built-in automatic checks**
5. **Options for all OS** (Linux/Mac/Windows)

This file can be placed at the root of your project and updated as dependencies evolve.

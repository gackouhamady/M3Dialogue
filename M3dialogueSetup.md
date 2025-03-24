# Configuration de l'environnement pour M3Dialogue

## 1. Prérequis système
- **GPU** : NVIDIA CUDA ≥ 11.3 (recommandé)
- **RAM** : ≥ 16GB (32GB pour les gros datasets)
- **Espace disque** : ≥ 50GB

## 2. Création de l'environnement virtuel
```bash
# Pour Linux/Mac
python -m venv m3dialogue_env
source m3dialogue_env/bin/activate

# Pour Windows
python -m venv m3dialogue_env
.\m3dialogue_env\Scripts\activate
```
**Note** : Conserver le terminal ouvert après activation

## 3. Installation des bibliothèques de base
```bash
# PyTorch avec support CUDA
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Bibliothèques essentielles
pip install transformers numpy pandas tqdm
```

## 4. Traitement multimodal

### Texte
```bash
pip install spacy nltk sentence-transformers
python -m spacy download en_core_web_sm  # Modèle anglais pour spaCy
```

### Audio
```bash
pip install librosa torchaudio pydub
```

### Visuel
```bash
pip install opencv-python face-alignment mediapipe
```

## 5. Graph Neural Networks
```bash
# DGL (Deep Graph Library)
pip install dgl-cu113 -f https://data.dgl.ai/wheels/cu113/repo.html
##  Pour CPU 
pip install dgl dglgo

# PyTorch Geometric
pip install torch-geometric
# Version CUDA 11.3 (GPU)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric

```

## 6. Évaluation et suivi
```bash
pip install scikit-learn wandb matplotlib seaborn
```

## 7. Vérification de l'installation
```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Version PyTorch: {torch.__version__}")
```

## 8. Fichier requirements.txt (optionnel)
```bash
# Générer le fichier
pip freeze > requirements.txt

# Réinstaller ultérieurement
pip install -r requirements.txt
```

## Bonnes pratiques
### Tester les imports critiques :
```python
from transformers import BertModel  # Test BERT
import dgl  # Test GNN
import librosa  # Test audio
```

### Configuration W&B :
```bash
wandb login  # Suivi des expériences
```

### Résolution des conflits :
```bash
pip check  # Vérifie les incompatibilités
```
**Warning** : Pour les problèmes CUDA, vérifiez la version avec `nvcc --version`

---

## Fonctionnalités du fichier :
1. **Sections claires** avec hiérarchie visuelle
2. **Commandes prêtes à copier-coller**
3. **Commentaires contextuels** pour chaque étape
4. **Vérifications automatiques** incluses
5. **Options pour tous les OS** (Linux/Mac/Windows)

Ce fichier peut être placé à la racine de votre projet et mis à jour au fur et à mesure de l'évolution des dépendances.



# M3Dialogue: Multimodal Dynamic Memory Network for Emotion Recognition in Conversations

## Introduction

Our proposed architecture, **M3Dialogue** (*Multimodal & Dynamic Memory for Dialogue Emotion Recognition*), represents a significant evolution beyond existing approaches like DialogueGCN and DialogueRNN by addressing three key limitations in current conversation-based emotion recognition systems:

1. **Static Context Modeling**: Traditional GCN-based approaches use fixed relation types
2. **Unimodal Limitations**: Most systems process only textual inputs
3. **Rigid Context Windows**: Fixed-size context windows fail to adapt to conversation dynamics

## Key Innovations Over Previous Architectures

| Feature               | DialogueRNN | DialogueGCN | M3Dialogue (Ours) |
|-----------------------|------------|------------|-------------------|
| **Modality Support**  | Text-only  | Text-only  | Text+Audio+Visual |
| **Relation Modeling** | Sequential | Fixed (2M²) | Dynamic Attention |
| **Context Window**    | Full history | Fixed window | Adaptive learning |
| **Memory Mechanism**  | GRU-based  | None       | Cross-modal Memory |

## Architecture

 ![Mon Image](dialoguegcn++_architecture_v0.png)













## Pipeline design of DGCN 
```bash
 =====================================================================
|                                                                   |
|  [Input Features] (seq_len, batch, D_m)                           |
|        |                                                          |
|        v                                                          |
|  +---------------------+                                          |
|  | BASE MODEL          |                                          |
|  | (DialogRNN/LSTM/GRU)|--> (seq_len, batch, 2*D_e)               |
|  +---------------------+                                          |
|        |                                                          |
|        v                                                          |
|  +---------------------+                                          |
|  | Adaptive Temporal   |                                          |
|  | Attention           |--> Edge weights + Dynamic Connections    |
|  +---------------------+                                          |
|        |                                                          |
|        v                                                          |
|  +---------------------+                                          |
|  | Graph Construction  |                                          |
|  | - edge_index        |--> (2, num_edges)                        |
|  | - edge_norm         |--> (num_edges,)                          |
|  | - edge_type         |--> (num_edges,)                          |
|  +---------------------+                                          |
|        |                                                          |
|        v                                                          |
|  +---------------------+                                          |
|  | Graph Network       |                                          |
|  | - RGCNConv          |--> (num_nodes, graph_hidden_size)        |
|  | - GraphConv         |                                          |
|  +---------------------+                                          |
|        |                                                          |
|        v                                                          |
|  +---------------------+                                          |
|  | Nodal Attention     | (Optionnel)                              |
|  | (MatchingAttention) |--> Features pondérées                   |
|  +---------------------+                                          |
|        |                                                          |
|        v                                                          |
|  +---------------------+                                          |
|  | Classification      |                                          |
|  | - Linear(D_h)       |--> (seq_len, batch, n_classes)           |
|  | - LogSoftmax        |                                          |
|  +---------------------+                                          |
|                                                                   |
=====================================================================
```

## Flux de Données Détaillé
### 1. Couche d'Entrée
- Format : (seq_len, batch, D_m)

- seq_len : Longueur de la séquence dialogue.

- batch : Taille du batch.

- D_m : Dimension des features (ex. embeddings textuels).

### 2. Encodage Séquentiel (Base Model)
- DialogRNN (si base_model='DialogRNN') :

- Passe avant (emotions_f) et arrière (emotions_b).

- Sortie concaténée : (seq_len, batch, 2*D_e).

- LSTM/GRU :

- Couches bidirectionnelles → Sortie de taille 2*D_e.

- None :

- Simple projection linéaire : Linear(D_m → 2*D_e).

### 3. Mécanisme d'Attention Temporelle
- AdaptiveTemporalAttention :

Calcule des scores d'attention pour les arêtes dynamiques.

- Prend en compte :

- max_seq_len pour l'encodage positionnel.

- window_past/window_future pour le fenêtrage initial.

- Sortie : Poids des arêtes (edge_norm).

### 4. Construction du Graphe
- batch_graphify :

- Nœuds : États du dialogue (features = sortie du base model).

- Arêtes :

- edge_index : Connexions (src, tgt) basées sur :

- Fenêtres temporelles (window_past/future).

- Rôles des interlocuteurs (qmask).

- edge_type : Type de relation (ex. "locuteur A → B").

- edge_norm : Poids des arêtes (via attention).

### 5. Graph Network
- RGCNConv :

- Prend en compte les types d'arêtes (edge_type).

- Normalisation par edge_norm.

- GraphConv :

- Agrège les informations des voisins.

### 6. Nodal Attention (Optionnel)
- MatchingAttention :

- Si nodal_attention=True, pondère les features des nœuds.

- Utilise un mécanisme general2 pour capturer les dépendances complexes.

### 7. Classification
- Couche Linéaire : Linear(2*D_e → D_h).

- Softmax : LogSoftmax pour les log-probabilités.

- Sortie : (seq_len, batch, n_classes).

- Exemple Visuel (Cas DialogRNN + GCN)
  ```bash
                   Input
                     |
          -------------------------
          |                       |
     DialogRNN (Forward)     DialogRNN (Backward)
          |                       |
          -----------+------------
                     |
              2*D_e Features
                     |
           Adaptive Temporal Attention
                     |
    +----------------+----------------+
    |                |                |
  edge_index      edge_norm        edge_type
    |                |                |
    +----------------+----------------+
                     |
               RGCN + GraphConv
                     |
           Nodal Attention (Optionnel)
                     |
                Classification
                     |
               Emotion Predictions

   ```
### Points Clés à Visualiser
- Bidirectionnalité : Les flèches avant/arrière pour DialogRNN ou LSTM/GRU.

- Graphe Dynamique :

- Nœuds = états de dialogue.

- Arêtes = relations (temporelles + interlocuteurs).

- Attention : Deux niveaux (temporel + nodal).

- Classification : Couche finale avec n_classes sorties.

 


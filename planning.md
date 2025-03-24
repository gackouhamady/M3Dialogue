# Planning d'Implémentation - M3Dialogue (Amélioration de GCN)

Voici un plan détaillé pour l'implémentation de  l' architecture avancée DialogueGCN++, organisé en phases claires avec livrables et dépendances :

| **Phase**                | **Tâche**                        | **Sous-Tâches**                                                                 | **Durée Estimée** | **Livrables**                                | **Dépendances** |
|--------------------------|----------------------------------|---------------------------------------------------------------------------------|-------------------|----------------------------------------------|-----------------|
| **1. Préparation**        | Configuration de l'environnement | - Installer PyTorch/TensorFlow<br>- Configurer CUDA si GPU<br>- Structurer le projet (modules/tests) | 2 jours           | Environnement fonctionnel<br>Structure de projet | Aucune          |
| **2. Inputs Multimodaux** | Intégration des données          | - Adapter les loaders pour MELD/IEMOCAP<br>- Prétraitement texte (Glove/BERT)<br>- Conversion audio → log-mel spectrograms<br>- Détection faciale (MTCNN) | 5 jours           | Pipeline de données unifié<br>Exemples prétraités | Phase 1         |
| **3. Extraction de Features** | Modules CNN spécialisés       | - Texte : CNN multi-filtres (3/4/5)<br>- Audio : Conv1D optimisé<br>- Visuel : Conv2D avec ResNet modifié | 7 jours           | Modules testés individuellement<br>Benchmarks de performance | Phase 2         |
| **4. Fusion Multimodale** | Stratégie de fusion              | - Concaténation intelligente<br>- Normalisation couche-specific<br>- Dropout adaptatif | 3 jours           | Features fusionnées (shape cohérente)        | Phase 3         |
| **5. Encodage Contextuel**| Bi-GRU + Fenêtre Adaptative      | - Implémenter Bi-GRU avec masques<br>- Mécanisme de fenêtre dynamique<br>- Gestion des padding sequences | 5 jours           | Contexte conversationnel encodé              | Phase 4         |
| **6. GCN Dynamique**      | Graph Relations avec Attention   | - Construction du graphe conversationnel<br>- Mécanisme d'attention relationnelle<br>- Couches GCN itératives | 10 jours          | Module GCN testé<br>Matrices d'attention visualisables | Phase 5         |
| **7. Classifieur**        | FFN + Softmax                    | - Fully Connected Layers<br>- Régularisation avancée (Label Smoothing)<br>- Optimisation Loss (Focal Loss) | 3 jours           | Probabilités d'émotions                     | Phase 6         |
| **8. Optimisation**       | Fine-tuning global               | - Hyperparamètre search (Optuna)<br>- Early Stopping<br>- Gradient Clipping | 7 jours           | Meilleur modèle sauvegardé                  | Phase 1-7       |
| **9. Évaluation**         | Benchmarks                       | - Comparaison avec GCN de base<br>- Métriques : F1, Accuracy, Confusion Matrix<br>- Analyse d'erreurs | 5 jours           | Rapport de performance<br>Visualisations    | Phase 8         |

## Feuille de Route Visuelle (Gantt Simplifié)

```mermaid
gantt
    title Timeline d'Implémentation
    dateFormat  YYYY-MM-DD
    section Préparation
    Environnement       :done, a1, 2025-03-24, 2d
    section Données
    Prétraitement      :active, a2, after a1, 5d
    section Modèle
    Extraction Features : a3, after a2, 7d
    Fusion             : a4, after a3, 3d
    Encodage           : a5, after a4, 5d
    GCN Dynamique      : a6, after a5, 10d
    Classifieur        : a7, after a6, 3d
    section Optimisation
    Fine-tuning        : a8, after a7, 7d
    Évaluation         : a9, after a8, 5d

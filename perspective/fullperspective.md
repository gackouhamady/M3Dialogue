# Perspectives d'Amélioration de DialogueGCN

## Attention Temporelle Adaptative
**Problème Identifié** : Les modèles RNN, y compris DialogueRNN, ont des problèmes de propagation d'informations à long terme, ce qui affecte l'efficacité de l'encodage du contexte séquentiel.

**Solution Proposée** : Intégrer un mécanisme d’attention temporelle adaptative pour améliorer la flexibilité du modèle et permettre une adaptation en temps réel aux changements conversationnels.

**Référence** : [Bradbury et al., 2017](https://arxiv.org/abs/1706.03762)

## Fenêtres Temporelles Adaptatives
**Problème Identifié** : La construction des graphes avec des fenêtres de contexte fixes peut être coûteuse en termes de calcul et limiter la capture du contexte pertinent.

**Solution Proposée** : Remplacer les fenêtres temporelles fixes par des fenêtres adaptatives, prédites en fonction du contexte, afin de réduire le biais des fenêtres fixes.

**Référence** : [Schlichtkrull et al., 2018](https://arxiv.org/abs/1703.06103)

## Renforcement Contextuel pour Mots Courts
**Problème Identifié** : Les courtes interventions comme "okay" ou "oui" dépendent fortement du contexte pour une classification émotionnelle précise.

**Solution Proposée** : Utiliser un renforcement contextuel pour les interventions courtes afin d'améliorer l'analyse de ces mots en les enrichissant avec le contexte global.

**Référence** : [Navarretta et al., 2016](https://link.springer.com/article/10.1007/s10579-016-9371-6)

## Intégration d'Apprentissage Contrastif
**Problème Identifié** : La différenciation des émotions similaires peut être difficile avec les méthodes actuelles.

**Solution Proposée** : Incorporer des techniques d'apprentissage contrastif pour améliorer la différenciation des émotions similaires et renforcer la robustesse du modèle.

**Référence** : [Chen et al., 2017](https://arxiv.org/pdf/2002.05709v1)

## Data Augmentation
**Problème Identifié** : Les jeux de données limités peuvent restreindre la performance et la généralisation du modèle.

**Solution Proposée** : Utiliser des techniques de data augmentation pour enrichir les jeux de données et améliorer la performance du modèle sur des données variées.

**Référence** : [Zadeh et al., 2018a](https://aclanthology.org/P18-1208/)

## Changement de Dataset
**Problème Identifié** : Tester le modèle sur différents jeux de données pour évaluer sa généralisation et sa robustesse.

**Solution Proposée** : Tester le modèle sur différents jeux de données pour évaluer sa généralisation et sa robustesse.

**Référence** : [Busso et al., 2008](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf)

## Architecture Hybride GCN + Transformer
**Problème Identifié** : Les GCN et les Transformers ont des avantages complémentaires qui peuvent être exploités ensemble.

**Solution Proposée** : Intégrer une architecture hybride combinant Graph Convolutional Networks (GCN) et Transformers pour bénéficier des avantages des deux approches.

**Référence** : [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

## Multimodalité
**Problème Identifié** : L'utilisation de données uniquement textuelles peut limiter la précision de la reconnaissance des émotions.

**Solution Proposée** : Incorporer des données multimodales (texte, audio, vidéo) pour enrichir la reconnaissance des émotions et améliorer la précision du modèle.

**Référence** : [Poria et al., 2017](https://aclanthology.org/D17-1115/)

## Résumé des Modifications

- **AdaptiveTemporalAttention** : Remplace MaskedEdgeAttention pour calculer des scores d’attention dynamiques.
- **dynamic_edge_perms** : Remplace edge_perms pour prédire la fenêtre temporelle optimale.
- **ContextEnhancer** : Renforce les mots courts avec le contexte global.

## Validation Expérimentale

- **Métriques à surveiller** : Accuracy sur les interventions courtes, F1-score pour les émotions ambiguës, temps d’entraînement.
- **Jeu de test** : Conversations avec nombreuses interventions courtes (DAIC-WOZ, MELD).

## Étapes Suivantes

- Implémenter les nouvelles classes.
- Benchmark sur MELD/IEMOCAP.
- Fine-tuning pour équilibrer performance/complexité.

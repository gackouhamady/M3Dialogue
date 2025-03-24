# Liste des Datasets pour la Reconnaissance d'Émotions dans les Conversations

## 1. Datasets Multimodaux (Texte + Audio + Visuel)

### MELD (Multimodal EmotionLines Dataset)
- **Description** : Conversations multipartites de la série *Friends*, annotées avec 7 émotions.
- **Taille** : 13k+ utterances, 1.4k dialogues.
- **Modalités** : Texte, audio, vidéo.
- **Lien** : [https://affective-meld.github.io](https://affective-meld.github.io)

### IEMOCAP (Interactive Emotional Dyadic Motion Capture)
- **Description** : Conversations dyadiques en studio avec 6 émotions annotées.
- **Taille** : 10k+ utterances.
- **Modalités** : Texte, audio, capture de mouvement.
- **Lien** : [https://sail.usc.edu/iemocap/](https://sail.usc.edu/iemocap/)

### CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity)
- **Description** : Vidéos YouTube avec émotions et intensité (valence/arousal).
- **Taille** : 23.5k+ clips.
- **Modalités** : Texte, audio, vidéo.
- **Lien** : [https://github.com/A2Zadeh/CMU-MOSEI](https://github.com/A2Zadeh/CMU-MOSEI)

---

## 2. Datasets Textuels (Conversations Annotées)

### EmoryNLP
- **Description** : Conversations de *Friends* avec émotions et relations sociales.
- **Taille** : 12k+ utterances.
- **Lien** : [https://github.com/emorynlp/emorynlp](https://github.com/emorynlp/emorynlp)

### DailyDialog
- **Description** : Conversations quotidiennes en anglais (7 émotions).
- **Taille** : 13k+ dialogues.
- **Lien** : [https://arxiv.org/abs/1710.03957](https://arxiv.org/abs/1710.03957)

### EmotionPush
- **Description** : Messages Facebook annotés avec 6 émotions.
- **Taille** : 1k+ conversations.
- **Lien** : [https://github.com/facebookresearch/EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues)

---

## 3. Datasets Audio-Visuel (Sans Texte)

### RAVDESS (Ryerson Audio-Visual Database)
- **Description** : Acteurs exprimant 8 émotions de base.
- **Taille** : 7.3k clips.
- **Modalités** : Audio + vidéo.
- **Lien** : [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

### CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)
- **Description** : Clips audio-visuels avec 6 émotions.
- **Taille** : 7.4k clips.
- **Lien** : [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

---

## 4. Datasets Géants (Pour Pré-entraînement)

### YouTube-8M Emotions
- **Description** : 7M vidéos YouTube avec valence/arousal.
- **Lien** : [https://research.google.com/youtube8m/](https://research.google.com/youtube8m/)

### AffectNet
- **Description** : 1M images faciales annotées (8 émotions).
- **Lien** : [http://mohammadmahoor.com/affectnet/](http://mohammadmahoor.com/affectnet/)

---

## Recommandations d'Utilisation
| Cas d'Usage                          | Dataset(s) Recommandé(s)  |
|--------------------------------------|---------------------------|
| **Conversations multimodales**       | MELD + IEMOCAP            |
| **Modèles lourds**                   | CMU-MOSEI + YouTube-8M    |
| **Conversations informelles**        | DailyDialog + EmotionPush |
| **Audio-Visuel seul**                | RAVDESS + CREMA-D         |

> **Note** : Pour les droits d'utilisation, vérifiez les licences spécifiques à chaque dataset.
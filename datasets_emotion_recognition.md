# List of Datasets for Emotion Recognition in Conversations

## 1. Multimodal Datasets (Text + Audio + Visual)

### MELD (Multimodal EmotionLines Dataset)
- **Description**: Multi-party conversations from the *Friends* series, annotated with 7 emotions.
- **Size**: 13k+ utterances, 1.4k dialogues.
- **Modalities**: Text, audio, video.
- **Link**: [https://affective-meld.github.io](https://affective-meld.github.io)

### IEMOCAP (Interactive Emotional Dyadic Motion Capture)
- **Description**: Dyadic conversations recorded in a studio with 6 annotated emotions.
- **Size**: 10k+ utterances.
- **Modalities**: Text, audio, motion capture.
- **Link**: [https://sail.usc.edu/iemocap/](https://sail.usc.edu/iemocap/)

### CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity)
- **Description**: YouTube videos annotated with emotions and intensity (valence/arousal).
- **Size**: 23.5k+ clips.
- **Modalities**: Text, audio, video.
- **Link**: [https://github.com/A2Zadeh/CMU-MOSEI](https://github.com/A2Zadeh/CMU-MOSEI)

---

## 2. Text-Based Datasets (Annotated Conversations)

### EmoryNLP
- **Description**: *Friends* conversations annotated with emotions and social relationships.
- **Size**: 12k+ utterances.
- **Link**: [https://github.com/emorynlp/emorynlp](https://github.com/emorynlp/emorynlp)

### DailyDialog
- **Description**: Daily English conversations (7 emotions).
- **Size**: 13k+ dialogues.
- **Link**: [https://arxiv.org/abs/1710.03957](https://arxiv.org/abs/1710.03957)

### EmotionPush
- **Description**: Facebook messages annotated with 6 emotions.
- **Size**: 1k+ conversations.
- **Link**: [https://github.com/facebookresearch/EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues)

---

## 3. Audio-Visual Datasets (Without Text)

### RAVDESS (Ryerson Audio-Visual Database)
- **Description**: Actors expressing 8 basic emotions.
- **Size**: 7.3k clips.
- **Modalities**: Audio + video.
- **Link**: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

### CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)
- **Description**: Audio-visual clips with 6 emotions.
- **Size**: 7.4k clips.
- **Link**: [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

---

## 4. Large-Scale Datasets (For Pretraining)

### YouTube-8M Emotions
- **Description**: 7M YouTube videos annotated with valence/arousal.
- **Link**: [https://research.google.com/youtube8m/](https://research.google.com/youtube8m/)

### AffectNet
- **Description**: 1M annotated facial images (8 emotions).
- **Link**: [http://mohammadmahoor.com/affectnet/](http://mohammadmahoor.com/affectnet/)

---

## Usage Recommendations
| Use Case                          | Recommended Dataset(s)  |
|-----------------------------------|-------------------------|
| **Multimodal conversations**      | MELD + IEMOCAP          |
| **Large-scale models**            | CMU-MOSEI + YouTube-8M  |
| **Informal conversations**        | DailyDialog + EmotionPush |
| **Audio-visual only**             | RAVDESS + CREMA-D       |

> **Note**: For usage rights, check the specific licenses of each dataset.

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

## Architectural Advantages

 ![Mon Image](dialoguegcn++_architecture_v0.png)

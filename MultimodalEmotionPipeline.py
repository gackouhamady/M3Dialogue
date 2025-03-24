"""
Multimodal Emotion Recognition Pipeline (DialogueGCN++ Compatible)
Intègre les méthodes des papiers fondateurs avec leurs paramètres originaux
"""

import os
import numpy as np
import torch
import librosa
import cv2
from torch.utils.data import Dataset
from transformers import BertTokenizer
from gensim.models import KeyedVectors
from mtcnn import MTCNN
from typing import Dict, Union, Optional

class MultimodalEmotionPipeline:
    """Pipeline complet pour le prétraitement des données multimodales (Texte/Audio/Visuel)
    conforme aux références académiques et optimisé pour DialogueGCN++
    
    Attributs:
        text_mode (str): 'glove' ou 'bert' (défaut: 'bert')
        device (str): 'cuda' ou 'cpu'
    """
    
    def __init__(self, 
                 text_mode: str = 'bert',
                 glove_path: Optional[str] = None,
                 device: str = 'cuda'):
        
        self.text_mode = text_mode
        self.device = device
        
        # Initialisation des composants
        self._init_text_processor(glove_path)
        self._init_audio_processor()
        self._init_face_processor()
    
    def _init_text_processor(self, glove_path: Optional[str]):
        """Initialise le processeur texte selon le mode sélectionné"""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        if self.text_mode == 'glove':
            if not glove_path:
                raise ValueError("Chemin vers les embeddings GloVe requis")
            self.glove = KeyedVectors.load_word2vec_format(glove_path)
    
    def _init_audio_processor(self):
        """Configure les paramètres audio suivant ICASSP 2017"""
        self.audio_params = {
            'sr': 16000,
            'n_fft': 400,
            'hop_length': 160,
            'n_mels': 80,
            'win_length': 400,
            'window': 'hann'
        }
    
    def _init_face_processor(self):
        """Initialise MTCNN avec paramètres originaux du papier"""
        self.face_detector = MTCNN(
            steps_threshold=[0.6, 0.7, 0.7],
            margin=44,
            device=self.device
        )
    
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Traite le texte selon BERT ou GloVe
        
        Args:
            text: Chaîne de caractères à traiter
            
        Returns:
            Dict avec 'embeddings' et 'attention_mask'
        """
        if self.text_mode == 'bert':
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=50,
                truncation=True
            )
            return {k: v.to(self.device) for k, v in inputs.items()}
        
        elif self.text_mode == 'glove':
            words = text.lower().split()
            embeddings = []
            for word in words:
                if word in self.glove:
                    embeddings.append(self.glove[word])
            
            if not embeddings:
                embeddings = [np.zeros(300)]
            
            mean_embedding = torch.FloatTensor(np.mean(embeddings, axis=0))
            return {
                'embeddings': mean_embedding.to(self.device),
                'attention_mask': torch.ones(1).to(self.device)
            }
    
    def process_audio(self, audio_path: str) -> torch.Tensor:
        """Convertit un fichier audio en log-mel spectrogramme
        
        Args:
            audio_path: Chemin vers le fichier audio (.wav)
            
        Returns:
            Tensor (n_mels, time_steps) normalisé
        """
        y, sr = librosa.load(audio_path, sr=self.audio_params['sr'])
        
        # STFT
        stft = librosa.stft(
            y,
            n_fft=self.audio_params['n_fft'],
            hop_length=self.audio_params['hop_length'],
            win_length=self.audio_params['win_length'],
            window=self.audio_params['window']
        )
        
        # Mel Spectrogram
        mel_basis = librosa.filters.mel(
            sr=self.audio_params['sr'],
            n_fft=self.audio_params['n_fft'],
            n_mels=self.audio_params['n_mels']
        )
        mel = np.dot(mel_basis, np.abs(stft)**2)
        
        # Log compression
        logmel = librosa.power_to_db(mel, ref=np.max)
        
        # Normalisation
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
        
        return torch.FloatTensor(logmel).to(self.device)
    
    def process_video_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Extrait et aligne un visage depuis une frame vidéo
        
        Args:
            frame: Image numpy (H, W, C)
            
        Returns:
            Tensor (3, 224, 224) normalisé [0,1]
        """
        # Détection et alignement
        results = self.face_detector.detect_faces(frame)
        
        if not results:
            return torch.zeros((3, 224, 224)).to(self.device)
            
        # Sélection meilleure détection
        best_face = max(results, key=lambda x: x['confidence'])
        x, y, w, h = best_face['box']
        
        # Alignement par landmarks
        keypoints = best_face['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Calcul angle rotation
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        # Rotation
        center = ((left_eye[0] + right_eye[0]) // 2, 
                 (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                               flags=cv2.INTER_CUBIC)
        
        # Crop et resize
        aligned_face = aligned[y:y+h, x:x+w]
        resized_face = cv2.resize(aligned_face, (224, 224))
        
        # Normalisation et permutation des canaux
        normalized = resized_face / 255.0
        tensor_face = torch.FloatTensor(normalized).permute(2, 0, 1)
        
        return tensor_face.to(self.device)
    
    def process_sample(self,
                      text: str,
                      audio_path: Optional[str] = None,
                      video_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Traite un échantillon complet multimodal
        
        Args:
            text: Transcription textuelle
            audio_path: Chemin vers fichier audio
            video_path: Chemin vers fichier vidéo
            
        Returns:
            Dictionnaire avec:
                - text_features: Sortie du processeur texte
                - audio_features: Spectrogramme (si audio_path fourni)
                - visual_features: Visage (si video_path fourni)
        """
        output = {
            'text_features': self.process_text(text)
        }
        
        if audio_path:
            output['audio_features'] = self.process_audio(audio_path)
            
        if video_path:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                output['visual_features'] = self.process_video_frame(frame)
            cap.release()
            
        return output


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    pipeline = MultimodalEmotionPipeline(
        text_mode='bert',
        glove_path='glove.6B.300d.txt' if 'glove' else None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Traitement d'un échantillon
    sample = pipeline.process_sample(
        text="I'm feeling excited!",
        audio_path="sample.wav",
        video_path="sample.mp4"
    )
    
    # Affichage des shapes
    print("\nFeatures extraites:")
    print(f"Texte (BERT): {sample['text_features']['input_ids'].shape}")
    if 'audio_features' in sample:
        print(f"Audio: {sample['audio_features'].shape}")
    if 'visual_features' in sample:
        print(f"Visuel: {sample['visual_features'].shape}")
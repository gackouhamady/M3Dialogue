import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd



class IEMOCAPDataset(Dataset):
    """
    Enhanced IEMOCAP Dataset loader for DialogueGCN++ architecture
    Handles multimodal inputs (text, audio, visual) with improved feature extraction
    and contextual encoding as shown in the new architecture.
    """
    
    def __init__(self, train=True):
        """
        Initialize the dataset with multimodal features
        
        Args:
            train (bool): Flag to load training or testing data
        """
        # Load multimodal features from pickle file
        (self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,
         self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,
         self.testVid) = pickle.load(open('./data/IEMOCAP_features.pkl', 'rb'), encoding='latin1')
        
        """
        Label index mapping:
        {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        """
        
        # Select appropriate video IDs based on train/test mode
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        """
        Get a single multimodal conversation sample with enhanced features
        
        Returns:
            text_features: GloVe embeddings for text (royalblue in architecture)
            visual_features: Facial images features (purple in architecture)
            audio_features: Log-Mel Spectrograms (orange in architecture)
            speaker_info: Speaker identity features
            mask: Sequence mask
            labels: Emotion labels
            vid: Video ID
        """
        vid = self.keys[index]
        
        # Extract features for each modality
        # Text features (will pass through CNN for text with fillers 3/4/5)
        text_features = self.videoText[vid]
        
        # Convert text features to tensor (if it's a list of numpy arrays or lists of lists)
        if isinstance(text_features, list):
            text_features = [torch.FloatTensor(feat) for feat in text_features]
            text_features = torch.stack(text_features)  # Shape should be (sequence_length, embedding_size)
        
        # Ensure text features are 2D (sequence_length x embedding_size)
        if text_features.ndimension() == 3:
            text_features = text_features.squeeze(0)  # Remove the extra batch dimension

        # Visual features (will pass through Conv2D)
        visual_features = torch.FloatTensor(self.videoVisual[vid])
        
        # Audio features (will pass through Conv1D)
        audio_features = torch.FloatTensor(self.videoAudio[vid])
        
        # Speaker information (one-hot encoded)
        speaker_info = torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]])
        
        # Mask for sequence
        mask = torch.FloatTensor([1]*len(self.videoLabels[vid]))
        
        # Emotion labels
        labels = torch.LongTensor(self.videoLabels[vid])
        
        return text_features, visual_features, audio_features, speaker_info, mask, labels, vid

    def __len__(self):
        """Return total number of conversations in dataset"""
        return self.len

    def collate_fn(self, data):
        """
        Custom collate function to handle variable length sequences
        and properly pad all modalities
        
        Args:
            data: List of samples from __getitem__
            
        Returns:
            Padded and batched tensors for each modality
        """
        dat = pd.DataFrame(data)
        return [
            pad_sequence(dat[i]) if i < 4 else  # Pad text, visual, audio, speaker features
            pad_sequence(dat[i], True) if i < 6 else  # Pad mask and labels
            dat[i].tolist()  # Keep video IDs as list
            for i in dat
        ]



class EnhancedAVECDataset(Dataset):
    """
    Enhanced AVEC dataset loader for DialogueGCN++ architecture
    Includes improvements for multimodal feature handling and contextual encoding
    """
    
    def __init__(self, path, train=True):
        """
        Initialize the dataset with path to features
        
        Args:
            path (str): Path to pickle file containing features
            train (bool): Flag to load training or testing data
        """
        (self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,
         self.videoAudio, self.videoVisual, self.videoSentence,
         self.trainVid, self.testVid) = pickle.load(open('./data/', 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        """
        Get a single sample with enhanced multimodal features
        
        Returns:
            Tuple containing all modalities and labels
        """
        vid = self.keys[index]
        return (
            torch.FloatTensor(self.videoText[vid]),  # Text features
            torch.FloatTensor(self.videoVisual[vid]),  # Visual features
            torch.FloatTensor(self.videoAudio[vid]),  # Audio features
            torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in self.videoSpeakers[vid]]),  # Speaker info
            torch.FloatTensor([1]*len(self.videoLabels[vid])),  # Mask
            torch.FloatTensor(self.videoLabels[vid])  # Labels (regression)
        )

    def __len__(self):
        """Return total number of conversations"""
        return self.len

    def collate_fn(self, data):
        """
        Custom collate function for AVEC dataset
        Handles padding for variable length sequences
        """
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) for i in dat]


class EnhancedMELDDataset(Dataset):
    """
    Enhanced MELD dataset loader for DialogueGCN++ architecture
    Supports both emotion and sentiment classification
    """
    
    def __init__(self, path, classify='emotion', train=True):
        """
        Initialize MELD dataset
        
        Args:
            path (str): Path to pickle file
            classify (str): 'emotion' or 'sentiment' classification
            train (bool): Flag for training or testing data
        """
        (self.videoIDs, self.videoSpeakers, self.videoLabelsEmotion, self.videoText,
         self.videoAudio, self.videoSentence, self.trainVid,
         self.testVid, self.videoLabelsSentiment) = pickle.load(open(path, 'rb'))

        self.classify = classify
        self.videoLabels = self.videoLabelsEmotion if classify == 'emotion' else self.videoLabelsSentiment
        
        """
        Emotion label index mapping:
        {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 
         'joy': 4, 'disgust': 5, 'anger':6}
        """
        
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        """
        Get a single sample with multimodal features
        
        Returns:
            Tuple containing text, audio, speaker info, mask, labels, and video ID
        """
        vid = self.keys[index]
        return (
            torch.FloatTensor(self.videoText[vid]),  # Text features
            torch.FloatTensor(self.videoAudio[vid]),  # Audio features
            torch.FloatTensor(self.videoSpeakers[vid]),  # Speaker info (already encoded)
            torch.FloatTensor([1]*len(self.videoLabels[vid])),  # Mask
            torch.LongTensor(self.videoLabels[vid]),  # Labels
            vid  # Video ID
        )

    def __len__(self):
        """Return total number of conversations"""
        return self.len

    def collate_fn(self, data):
        """
        Custom collate function for MELD dataset
        Handles padding for variable length sequences
        """
        dat = pd.DataFrame(data)
        return [
            pad_sequence(dat[i]) if i<3 else  # Pad text, audio, speaker features
            pad_sequence(dat[i], True) if i<5 else  # Pad mask and labels
            dat[i].tolist()  # Keep video IDs as list
            for i in dat
        ]


class EnhancedDailyDialogueDataset(Dataset):
    """
    Enhanced DailyDialogue dataset loader for DialogueGCN++
    Improved to handle the new architecture components
    """
    
    def __init__(self, split, path):
        """
        Initialize DailyDialogue dataset
        
        Args:
            split (str): 'train', 'test', or 'valid'
            path (str): Path to pickle file
        """
        (self.Speakers, self.Features, 
         self.ActLabels, self.EmotionLabels, 
         self.trainId, self.testId, self.validId) = pickle.load(open(path, 'rb'))
        
        # Select appropriate conversation IDs based on split
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        """
        Get a single conversation sample
        
        Returns:
            Tuple containing features, speaker info, mask, labels, and conversation ID
        """
        conv = self.keys[index]
        
        return (
            torch.FloatTensor(self.Features[conv]),  # Combined features
            torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),  # Speaker info
            torch.FloatTensor([1]*len(self.EmotionLabels[conv])),  # Mask
            torch.LongTensor(self.EmotionLabels[conv]),  # Labels
            conv  # Conversation ID
        )

    def __len__(self):
        """Return total number of conversations"""
        return self.len
    
    def collate_fn(self, data):
        """
        Custom collate function for DailyDialogue
        Handles padding for variable length sequences
        """
        dat = pd.DataFrame(data)
        return [
            pad_sequence(dat[i]) if i<2 else  # Pad features and speaker info
            pad_sequence(dat[i], True) if i<4 else  # Pad mask and labels
            dat[i].tolist()  # Keep conversation IDs as list
            for i in dat
        ]
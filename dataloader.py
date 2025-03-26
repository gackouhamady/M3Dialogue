import torch # for tensor and NN operations
from torch.utils.data import Dataset  # For Custom  data management 
from torch.nn.utils.rnn import pad_sequence  #  For padding sequence 
import pickle, pandas as pd # for serializing,  data mainipulation
import numpy as np  # Mathematicls  , matrixs computations

class IEMOCAPDataset(Dataset):
    def __init__(self, train=True):
        (self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,
         self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,
         self.testVid) = pickle.load(open('./data/IEMOCAP_features.pkl', 'rb'), encoding='latin1')
        
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        # Convert lists of numpy arrays into single numpy arrays before tensor conversion
        text_features = torch.tensor(np.array(self.videoText[vid]), dtype=torch.float32)
        visual_features = torch.tensor(np.array(self.videoVisual[vid]), dtype=torch.float32)
        audio_features = torch.tensor(np.array(self.videoAudio[vid]), dtype=torch.float32)

        speaker_info = torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]])
        mask = torch.FloatTensor([1]*len(self.videoLabels[vid]))
        labels = torch.LongTensor(self.videoLabels[vid])
        
        return text_features, visual_features, audio_features, speaker_info, mask, labels, vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [
            pad_sequence(dat[i]) if i < 4 else 
            pad_sequence(dat[i], True) if i < 6 else 
            dat[i].tolist()
            for i in dat
        ]

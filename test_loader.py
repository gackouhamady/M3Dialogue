import pytest
import torch
from torch.utils.data import DataLoader
from dataloader import IEMOCAPDataset  # Importez votre classe Dataset

# Fonction de test pour le dataloader
def test_dataloader():
    dataset = IEMOCAPDataset(train=True)  # Charge les données d'entraînement
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    # Assurez-vous que le dataset n'est pas vide
    assert len(dataset) > 0, "Dataset is empty!"

    # Test d'un seul batch
    for batch in dataloader:
        text_features, visual_features, audio_features, speaker_info, mask, labels, vid = batch

        # Vérifiez que les entrées sont des tensors
        assert isinstance(text_features, torch.Tensor), "Text features should be a tensor"
        assert isinstance(visual_features, torch.Tensor), "Visual features should be a tensor"
        assert isinstance(audio_features, torch.Tensor), "Audio features should be a tensor"
        assert isinstance(speaker_info, torch.Tensor), "Speaker info should be a tensor"
        assert isinstance(mask, torch.Tensor), "Mask should be a tensor"
        assert isinstance(labels, torch.Tensor), "Labels should be a tensor"

        # Vérifiez les dimensions des tenseurs
        assert text_features.ndimension() == 3, "Text features should be a 3D tensor"
        assert visual_features.ndimension() == 3, "Visual features should be a 3D tensor"
        assert audio_features.ndimension() == 3, "Audio features should be a 3D tensor"
        assert speaker_info.ndimension() == 3, "Speaker info should be a 3D tensor"
        assert mask.ndimension() == 2, "Mask should be a 2D tensor"
        assert labels.ndimension() == 2, "Labels should be a 2D tensor"
        
        # Assurez-vous qu'il y a des données dans les vidéos
        assert len(vid) > 0, "Video IDs should not be empty"
        
        break  # Tester seulement un batch

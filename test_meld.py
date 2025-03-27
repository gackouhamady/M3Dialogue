from collections import defaultdict
import pytest
import numpy as np
from dataloader_meld import Dataloader  # Assuming the file is named dataloader.py
import os
import pickle

# Fixture to create a test instance of Dataloader
@pytest.fixture
def sentiment_dataloader():
    # Create a small test pickle file if it doesn't exist
    test_data_path = "./data/data_sentiment.p"
    if not os.path.exists(test_data_path):
        os.makedirs("./data", exist_ok=True)
        test_data = [
            [  # revs
                {"text": "this is good", "split": "train", "y": "positive", "dialog": "d1", "utterance": "1", "num_words": 3},
                {"text": "this is bad", "split": "val", "y": "negative", "dialog": "d2", "utterance": "1", "num_words": 3},
                {"text": "neutral text", "split": "test", "y": "neutral", "dialog": "d3", "utterance": "1", "num_words": 2}
            ],
            np.random.rand(10, 300),  # W (embedding matrix)
            {"this": 0, "is": 1, "good": 2, "bad": 3, "neutral": 4, "text": 5},  # word_idx_map
            ["this", "is", "good", "bad", "neutral", "text"],  # vocab
            None,  # (unused in original code)
            {"positive": 0, "negative": 1, "neutral": 2}  # label_index
        ]
        with open(test_data_path, "wb") as f:
            pickle.dump(test_data, f)
    
    return Dataloader(mode="Sentiment")

def test_dataloader_initialization(sentiment_dataloader):
    assert sentiment_dataloader.MODE == "Sentiment"
    assert sentiment_dataloader.max_l == 50
    assert sentiment_dataloader.num_classes == 3
    assert isinstance(sentiment_dataloader.W, np.ndarray)
    assert isinstance(sentiment_dataloader.word_idx_map, dict)
    # Update to check for defaultdict instead of list
    from collections import defaultdict
    assert isinstance(sentiment_dataloader.vocab, defaultdict)

def test_get_word_indices(sentiment_dataloader):
    test_sentence = "this is good"
    result = sentiment_dataloader.get_word_indices(test_sentence)
    assert len(result) == 50
    # Replace [0, 1, 2] with the actual indices from your word_idx_map
    assert np.array_equal(result[:3], [21, 8, 90])  # Use your real indices

def test_get_dialogue_ids(sentiment_dataloader: Dataloader):
    test_keys = ["d1_1", "d1_2", "d2_1", "d3_1", "d3_2", "d3_3"]
    result = sentiment_dataloader.get_dialogue_ids(test_keys)
    assert isinstance(result, defaultdict)
    assert set(result.keys()) == {"d1", "d2", "d3"}
    assert result["d1"] == ["1", "2"]
    assert result["d3"] == ["1", "2", "3"]

def test_get_one_hot(sentiment_dataloader: Dataloader):
    # Test with label 0 (positive)
    assert sentiment_dataloader.get_one_hot(0) == [1, 0, 0]
    # Test with label 1 (negative)
    assert sentiment_dataloader.get_one_hot(1) == [0, 1, 0]
    # Test with label 2 (neutral)
    assert sentiment_dataloader.get_one_hot(2) == [0, 0, 1]

def test_data_splits(sentiment_dataloader: Dataloader):
    assert len(sentiment_dataloader.train_data) > 0
    assert len(sentiment_dataloader.val_data) > 0
    assert len(sentiment_dataloader.test_data) > 0
    
    # Check that utterance IDs are correctly formed
    for utt_id in sentiment_dataloader.train_data:
        assert "_" in utt_id
        dialog, utterance = utt_id.split("_")
        assert isinstance(dialog, str)
        assert utterance.isdigit()

def test_load_text_data(sentiment_dataloader: Dataloader):
    sentiment_dataloader.load_text_data()
    
    # Check that features were loaded correctly
    assert hasattr(sentiment_dataloader, "train_dialogue_features")
    assert hasattr(sentiment_dataloader, "val_dialogue_features")
    assert hasattr(sentiment_dataloader, "test_dialogue_features")
    
    # Check shapes
    assert sentiment_dataloader.train_dialogue_features.ndim == 3
    assert sentiment_dataloader.val_dialogue_features.ndim == 3
    assert sentiment_dataloader.test_dialogue_features.ndim == 3

def test_get_max_utts(sentiment_dataloader: Dataloader):
    # Create test dialogue IDs
    train_ids = {"d1": ["1", "2", "3"], "d2": ["1"]}
    val_ids = {"d3": ["1", "2"]}
    test_ids = {"d4": ["1"]}
    
    max_utts = sentiment_dataloader.get_max_utts(train_ids, val_ids, test_ids)
    assert max_utts == 3

def test_get_masks(sentiment_dataloader: Dataloader):
    # Set up test dialogue lengths
    sentiment_dataloader.train_dialogue_length = [3, 1, 2]
    sentiment_dataloader.val_dialogue_length = [2, 4]
    sentiment_dataloader.test_dialogue_length = [1]
    
    sentiment_dataloader.max_utts = 4
    sentiment_dataloader.get_masks()
    
    # Check train mask
    assert sentiment_dataloader.train_mask.shape == (3, 4)
    assert np.array_equal(sentiment_dataloader.train_mask[0], [1, 1, 1, 0])
    assert np.array_equal(sentiment_dataloader.train_mask[1], [1, 0, 0, 0])
    
    # Check val mask
    assert sentiment_dataloader.val_mask.shape == (2, 4)
    assert np.array_equal(sentiment_dataloader.val_mask[1], [1, 1, 1, 1])
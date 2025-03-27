import pytest
import torch
import numpy as np
from model_meld import MELDDialogueGCN, MaskedEdgeAttention, GraphNetwork

@pytest.fixture
def setup_model():
    # Setup model with small dimensions for testing
    model = MELDDialogueGCN(
        D_m_text=50,
        D_m_audio=20,
        D_m_visual=20,
        D_e=30,
        D_h=20,
        n_classes=7,
        max_seq_len=5,  # Changed to match test sequence length
        window_past=3,
        window_future=3,
        no_cuda=True
    )
    return model

@pytest.fixture
def test_data():
    # Create test data with batch size 2, sequence length 5
    seq_len = 5
    batch_size = 2
    
    # Random features (text, audio, visual)
    U_text = torch.randn(seq_len, batch_size, 50)
    U_audio = torch.randn(seq_len, batch_size, 20)
    U_visual = torch.randn(seq_len, batch_size, 20)
    
    # Speaker masks (2 speakers)
    qmask = torch.zeros(seq_len, batch_size, 2)
    qmask[:, 0, 0] = 1  # First sample: speaker 0
    qmask[:, 1, 1] = 1  # Second sample: speaker 1
    
    # Utterance masks (all valid)
    umask = torch.ones(batch_size, seq_len)
    
    # Sequence lengths
    seq_lengths = [seq_len, seq_len-1]  # Second sequence is shorter
    
    return U_text, U_audio, U_visual, qmask, umask, seq_lengths
"""
def test_model_initialization(setup_model):
    model = setup_model
    
    # Test model components exist
    assert hasattr(model, 'feature_fusion')
    assert hasattr(model, 'att_model')
    assert hasattr(model, 'graph_net')
    
    # Test dimensions
    assert model.feature_fusion.in_features == 90  # 50+20+20
    assert model.feature_fusion.out_features == 60  # 2*D_e (30)

def test_forward_pass(setup_model, test_data):
    model = setup_model
    U_text, U_audio, U_visual, qmask, umask, seq_lengths = test_data
    
    # Test forward pass
    outputs = model(U_text, U_audio, U_visual, qmask, umask, seq_lengths)
    log_prob, edge_index, edge_norm, edge_type, edge_index_lengths = outputs
    
    # Check output types and shapes
    assert isinstance(log_prob, torch.Tensor)
    
    # The output should be either:
    # - Per-node predictions (sum(seq_lengths) × n_classes)
    # - Per-sequence predictions (batch_size × n_classes)
    assert log_prob.shape[1] == 7  # n_classes
    assert log_prob.shape[0] in [sum(seq_lengths), len(seq_lengths)]
    
    # Check edge information
    assert edge_index.shape[0] == 2  # from and to nodes
    assert len(edge_norm) == edge_index.shape[1]
    assert len(edge_type) == edge_index.shape[1]
    assert len(edge_index_lengths) == 2  # batch size
"""
def test_masked_edge_attention():
    # Test MaskedEdgeAttention module with proper dimensions
    input_dim = 64
    max_seq_len = 5  # Matches test sequence length
    batch_size = 2
    
    att = MaskedEdgeAttention(input_dim=input_dim, max_seq_len=max_seq_len, no_cuda=True)
    
    # Create test input (seq_len, batch, input_dim)
    M = torch.randn(max_seq_len, batch_size, input_dim)
    lengths = [max_seq_len, max_seq_len-1]  # Second sequence shorter
    edge_ind = [[(0,1), (1,0), (1,2)], [(0,1), (1,2)]]  # Sample edges
    
    scores = att(M, lengths, edge_ind)
    
    # Check output shape (batch, max_seq_len, seq_len)
    assert scores.shape == (batch_size, max_seq_len, max_seq_len)
    assert not torch.isnan(scores).any()

def test_graph_network():
    # Test GraphNetwork module with corrected dimensions
    num_features = 128
    num_classes = 7
    num_relations = 8
    max_seq_len = 5  # Matches test sequence length
    
    gn = GraphNetwork(
        num_features=num_features,
        num_classes=num_classes,
        num_relations=num_relations,
        max_seq_len=max_seq_len,
        hidden_size=64,
        dropout=0.5,
        no_cuda=True
    )
    
    # Create test data with proper dimensions
    num_nodes = 8  # 5 + 3 nodes for two sequences
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0,1,1,2,3,4], [1,0,2,1,4,3]], dtype=torch.long)
    edge_norm = torch.rand(6)
    edge_type = torch.tensor([0,1,2,3,4,5], dtype=torch.long)
    seq_lengths = [5, 3]
    umask = torch.ones(2, max_seq_len)  # batch=2, max_seq_len=5
    
    # Test with nodal attention
    log_prob = gn(x, edge_index, edge_norm, edge_type, seq_lengths, umask, True, False)
    assert log_prob.shape[0] == num_nodes
    assert log_prob.shape[1] == num_classes
    
    # Test without nodal attention
    log_prob = gn(x, edge_index, edge_norm, edge_type, seq_lengths, umask, False, False)
    assert log_prob.shape[0] == num_nodes
    assert log_prob.shape[1] == num_classes

def test_edge_perms():
    from model_meld import edge_perms
    
    # Test with window_past=1, window_future=1
    perms = edge_perms(5, 1, 1)
    
    expected_perms = {
        (0,0), (0,1),
        (1,0), (1,1), (1,2),
        (2,1), (2,2), (2,3),
        (3,2), (3,3), (3,4),
        (4,3), (4,4)
    }
    
    assert set(perms) == expected_perms
    
    # Test with unlimited window
    perms = edge_perms(3, -1, -1)
    assert len(perms) == 9  # all possible pairs for seq_len=3

def test_batch_graphify(setup_model, test_data):
    from model_meld import batch_graphify
    
    _, _, _, qmask, _, seq_lengths = test_data
    model = setup_model
    
    # Create test features (output from feature fusion)
    features = torch.randn(5, 2, 60)  # seq_len=5, batch=2, dim=2*D_e
    
    # Test graph construction
    outputs = batch_graphify(
        features, qmask, seq_lengths,
        window_past=2, window_future=2,
        edge_type_mapping=model.edge_type_mapping,
        att_model=model.att_model,
        no_cuda=True
    )
    
    node_features, edge_index, edge_norm, edge_type, edge_index_lengths = outputs
    
    # Check outputs
    assert node_features.shape[0] == sum(seq_lengths)
    assert edge_index.shape[0] == 2
    assert len(edge_norm) == edge_index.shape[1]
    assert len(edge_type) == edge_index.shape[1]
    assert len(edge_index_lengths) == 2

def test_reverse_seq(setup_model, test_data):
    model = setup_model
    U_text, _, _, _, umask, _ = test_data
    
    # Test sequence reversal
    rev_U = model._reverse_seq(U_text, umask)
    
    assert rev_U.shape == U_text.shape
    # Check that the reversed sequence is actually reversed
    assert torch.allclose(U_text[0,0], rev_U[-1,0], atol=1e-6)
    assert torch.allclose(U_text[-1,0], rev_U[0,0], atol=1e-6)
"""
def test_model_output_range(setup_model, test_data):
    model = setup_model
    U_text, U_audio, U_visual, qmask, umask, seq_lengths = test_data
    
    # Test output is valid probability distribution
    log_prob, _, _, _, _ = model(U_text, U_audio, U_visual, qmask, umask, seq_lengths)
    prob = torch.exp(log_prob)
    
    # Check probabilities sum to ~1 (with tolerance for numerical errors)
    prob_sums = prob.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)
    
    # Check all probabilities between 0 and 1 (with small tolerance)
    assert (prob >= -1e-6).all() and (prob <= 1+1e-6).all() """
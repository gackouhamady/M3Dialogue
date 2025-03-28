import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np



class MatchingAttention(nn.Module):
    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general2'):
        super(MatchingAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        

    def forward(self, M, x, mask=None):
        if mask is None:
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        M_ = M.permute(1,2,0)
        x_ = self.transform(x).unsqueeze(1)
        mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
        M_ = M_ * mask_
        alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
        alpha_ = torch.tanh(alpha_)
        alpha_ = F.softmax(alpha_, dim=2)
        alpha_masked = alpha_*mask.unsqueeze(1)
        alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
        alpha = alpha_masked/alpha_sum

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha

class MELDDialogueGCN(nn.Module):
    def __init__(self, D_m_text=100, D_m_audio=100, D_m_visual=100, 
                 D_g=150, D_p=150, D_e=100, D_h=100, D_a=100, 
                 graph_hidden_size=64, n_speakers=2, max_seq_len=110,
                 window_past=10, window_future=10, n_classes=7,
                 dropout=0.5, dropout_rec=0.5, nodal_attention=True,
                 no_cuda=False):
        
        super(MELDDialogueGCN, self).__init__()
        
        self.D_m_text = D_m_text
        self.D_m_audio = D_m_audio
        self.D_m_visual = D_m_visual
        self.D_m = D_m_text + D_m_audio + D_m_visual
        self.n_classes = n_classes
        self.no_cuda = no_cuda
        self.nodal_attention = nodal_attention
        
        # Feature fusion layer
        self.feature_fusion = nn.Linear(self.D_m, 2*D_e)
        
        # Edge attention model
        self.att_model = MaskedEdgeAttention(2*D_e, max_seq_len, self.no_cuda)
        
        self.window_past = window_past
        self.window_future = window_future
        
        self.edge_type_mapping = {
            '000': 0, '001': 1,
            '010': 2, '011': 3,
            '100': 4, '101': 5,
            '110': 6, '111': 7
        }
        n_relations = len(self.edge_type_mapping)
        
        self.graph_net = GraphNetwork(2*D_e, n_classes, n_relations, 
                                    max_seq_len, graph_hidden_size, 
                                    dropout, self.no_cuda)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout_rec)
        
    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()
        
        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)
            
        return pad_sequence(xfs)
    
    def forward(self, U_text, U_audio, U_visual, qmask, umask, seq_lengths):
        # Feature fusion
        U = torch.cat([U_text, U_audio, U_visual], dim=-1)
        U = self.feature_fusion(U)
        U = self.dropout_rec(U)
        
        # Create forward and backward sequences
        emotions_f = U
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b = self._reverse_seq(rev_U, umask)
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        
        # Construct graph
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            emotions, qmask, seq_lengths, 
            self.window_past, self.window_future,
            self.edge_type_mapping, self.att_model, self.no_cuda
        )
        
        # Graph network forward pass
        log_prob = self.graph_net(
            features, edge_index, edge_norm, edge_type, 
            seq_lengths, umask, self.nodal_attention, False
        )
        
        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths

class MaskedEdgeAttention(nn.Module):
    def __init__(self, input_dim, max_seq_len, no_cuda):
        super(MaskedEdgeAttention, self).__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.no_cuda = no_cuda
        
    def forward(self, M, lengths, edge_ind):
        # Input shape: (seq_len, batch, input_dim)
        scale = self.scalar(M)  # shape: (seq_len, batch, max_seq_len)
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # shape: (batch, max_seq_len, seq_len)
        
        device = M.device
        mask = torch.ones_like(alpha) * 1e-10
        mask_copy = torch.zeros_like(alpha)
        mask, mask_copy = mask.to(device), mask_copy.to(device)
        
        edge_ind_ = []
        for i, j in enumerate(edge_ind):
            for x in j:
                edge_ind_.append([i, x[0], x[1]])
                
        if len(edge_ind_) > 0:
            edge_ind_ = torch.tensor(edge_ind_, dtype=torch.long, device=device).T
            edge_ind_[edge_ind_ >= alpha.shape[-1]] = alpha.shape[-1] - 1
            mask[edge_ind_[0], edge_ind_[1], edge_ind_[2]] = 1
            mask_copy[edge_ind_[0], edge_ind_[1], edge_ind_[2]] = 1
            
        masked_alpha = alpha * mask
        _sums = masked_alpha.sum(-1, keepdim=True)
        scores = masked_alpha.div(_sums + 1e-10) * mask_copy
        
        return scores  # shape: (batch, max_seq_len, seq_len)

class GraphNetwork(nn.Module):
    def __init__(self, num_features, num_classes, num_relations, 
                max_seq_len, hidden_size=64, dropout=0.5, no_cuda=False):
        super(GraphNetwork, self).__init__()
        
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.matchatt = MatchingAttention(num_features+hidden_size, num_features+hidden_size)
        self.linear = nn.Linear(num_features+hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.smax_fc = nn.Linear(hidden_size, num_classes)
        self.no_cuda = no_cuda
        
    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        if edge_index.size(1) > 0:
            out = self.conv1(x, edge_index, edge_type)
            out = self.conv2(out, edge_index)
        else:
            out = torch.zeros(x.size(0), self.conv2.out_channels, device=x.device)
        
        emotions = torch.cat([x, out], dim=-1)
        
        if nodal_attn:
            emotions = attentive_node_features(emotions, seq_lengths, umask, self.matchatt, self.no_cuda)
            hidden = F.relu(self.linear(emotions))
            hidden = self.dropout(hidden)
            hidden = self.smax_fc(hidden)
            
            if avec:
                return torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
            
            log_prob = F.log_softmax(hidden, 2)
            log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
            return log_prob
        else:
            hidden = F.relu(self.linear(emotions))
            hidden = self.dropout(hidden)
            hidden = self.smax_fc(hidden)
            
            if avec:
                return hidden
                
            log_prob = F.log_softmax(hidden, 1)
            return log_prob

def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []
    
    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))
    
    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j]
        edge_index_lengths.append(len(perms1))
        
        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])
            
            speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()
            
            direction = '0' if item1[0] < item1[1] else '1'
            edge_key = f"{speaker0}{speaker1}{direction}"
            edge_type.append(edge_type_mapping[edge_key])
    
    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1) if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_norm = torch.stack(edge_norm) if edge_norm else torch.empty((0,), dtype=torch.float)
    edge_type = torch.tensor(edge_type, dtype=torch.long) if edge_type else torch.empty((0,), dtype=torch.long)
    
    if not no_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_norm = edge_norm.to(device)
        edge_type = edge_type.to(device)
    
    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths

def edge_perms(l, window_past, window_future):
    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()
        
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)

def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda):
    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()
    
    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda) 
                          for s, l in zip(start.data.tolist(),
                          input_conversation_length.data.tolist())], 0).transpose(0, 1)

    att_emotions = []
    alpha = []

    for t in emotions:
        att_em, alpha_ = matchatt_layer(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:,0,:])

    att_emotions = torch.cat(att_emotions, dim=0)
    return att_emotions

def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor
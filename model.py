import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np, itertools, random, copy, math

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1,1)
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)/torch.sum(self.weight[target]*mask_.squeeze())
        return loss

class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1,2,0)
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha

class MatchingAttention(nn.Module):
    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        elif att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            M_ = M.permute(1,2,0)
            x_ = x.unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general2':
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
        else:
            M_ = M.transpose(0,1)
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1)
            M_x_ = torch.cat([M_,x_],2)
            mx_a = F.tanh(self.transform(M_x_))
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2)

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha

class DialogueRNNCell(nn.Module):
    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.listener_state = listener_state
        
        self.g_cell = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell = nn.GRUCell(D_p,D_e)
        
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p,D_p)

        self.dropout = nn.Dropout(dropout)
        
        if context_attention=='simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, e0):
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U,q0_sel], dim=1),
                torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0 else
                g_hist[-1])
        g_ = self.dropout(g_)
        
        if g_hist.size()[0]==0:
            c_ = torch.zeros(U.size()[0],self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist,U)
            
        U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = self.p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
            U_ss_ = torch.cat([U_,ss_],1)
            ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
            
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0 else e0
        e_ = self.e_cell(self._select_parties(q_,qm_idx), e0)
        e_ = self.dropout(e_)
        
        return g_,q_,e_,alpha

class DialogueRNN(nn.Module):
    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNN, self).__init__()
        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)
        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e,
                          listener_state, context_attention, D_a, dropout)

    def forward(self, U, qmask):
        g_hist = torch.zeros(0).type(U.type())
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_p).type(U.type())
        e_ = torch.zeros(0).type(U.type())
        e = e_
        alpha = []
        
        for u_,qmask_ in zip(U, qmask):
            g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
            e = torch.cat([e, e_.unsqueeze(0)],0)
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        return e,alpha

class MaskedEdgeAttention(nn.Module):
    def __init__(self, input_dim, max_seq_len, no_cuda):
        super(MaskedEdgeAttention, self).__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
        
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
        return scores

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
        
            if item1[0] < item1[1]:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])
    
    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()
    
    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths

class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations, max_seq_len, hidden_size=64, dropout=0.5, no_cuda=False):
        super(GraphNetwork, self).__init__()
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.matchatt = MatchingAttention(num_features+hidden_size, num_features+hidden_size, att_type='general2')
        self.linear = nn.Linear(num_features+hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.smax_fc = nn.Linear(hidden_size, num_classes)
        self.no_cuda = no_cuda 

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        out = self.conv1(x, edge_index, edge_type)
        out = self.conv2(out, edge_index)
        emotions = torch.cat([x, out], dim=-1)
        
        if nodal_attn:
            input_conversation_length = torch.tensor(seq_lengths)
            start_zero = input_conversation_length.data.new(1).zero_()
            
            if not self.no_cuda:
                input_conversation_length = input_conversation_length.cuda()
                start_zero = start_zero.cuda()

            max_len = max(seq_lengths)
            start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)
            emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, self.no_cuda) 
                                for s, l in zip(start.data.tolist(),
                                input_conversation_length.data.tolist())], 0).transpose(0, 1)

            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])

            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
            
        hidden = self.dropout(hidden)
        hidden = self.smax_fc(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        return log_prob

class DialogueGCNModel(nn.Module):
    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a, graph_hidden_size, n_speakers, max_seq_len, window_past, window_future,
                 n_classes=7, listener_state=False, context_attention='simple', dropout_rec=0.5, dropout=0.5, nodal_attention=True, avec=False, no_cuda=False):
        super(DialogueGCNModel, self).__init__()
        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda

        if self.base_model == 'DialogRNN':
            self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)
            self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)
        elif self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)
        else:
            print('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError 

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future
        self.att_model = MaskedEdgeAttention(2*D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention
        self.graph_net = GraphNetwork(2*D_e, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout, self.no_cuda)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)
        self.edge_type_mapping = edge_type_mapping

    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()
        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)
        return pad_sequence(xfs)

    def forward(self, U, qmask, umask, seq_lengths):
        if self.base_model == "DialogRNN":
            if self.avec:
                emotions, _ = self.dialog_rnn_f(U, qmask)
            else:
                emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)
                rev_U = self._reverse_seq(U, umask)
                rev_qmask = self._reverse_seq(qmask, umask)
                emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
                emotions_b = self._reverse_seq(emotions_b, umask)
                emotions = torch.cat([emotions_f,emotions_b],dim=-1)
        elif self.base_model == 'LSTM':
            emotions, hidden = self.lstm(U)
        elif self.base_model == 'GRU':
            emotions, hidden = self.gru(U)
        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions, qmask, seq_lengths, 
            self.window_past, self.window_future, self.edge_type_mapping, self.att_model, self.no_cuda)
            
        log_prob = self.graph_net(features, edge_index, edge_norm, edge_type, seq_lengths, umask, self.nodal_attention, self.avec)
        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths
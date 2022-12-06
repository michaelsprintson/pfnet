import torch.nn as nn
import torch
from pfrnns import PFLSTMCell
import numpy as np
from torch.utils.data.dataset import Dataset

class Tracker(nn.Module):
    """
    Implements PF-RNN structure for given batch and arguments
    """
    def __init__(self, args):
        """
        Initialize RNN with input arguments and start the PFLSTM cell
        """
        super(Tracker, self).__init__()
        self.num_particles = args.num_particles
        self.batch_size = args.batch_size
        self.window_length = args.window_length
        self.hidden_dim = args.h
        self.act_emb = args.emb_act
        self.dropout_rate = args.dropout
        self.num_obs = args.obs_num

        self.rnn = PFLSTMCell(self.num_particles, self.act_emb,
            self.hidden_dim, 32, 32, args.resamp_alpha)

        self.hidden2label = nn.Linear(self.hidden_dim, 1)

        self.act_embedding = nn.Linear(args.window_length*5, self.act_emb)
        self.hnn_dropout = nn.Dropout(self.dropout_rate)

        self.initialize = 'rand'
        self.args = args
        self.bp_length = args.bp_length

    def init_hidden(self, batch_size):
        """
        Initializes hidden states, cell states, and weights 
        """ 
        initializer = torch.rand if self.initialize == 'rand' else torch.zeros

        h0 = initializer(batch_size * self.num_particles, self.hidden_dim)
        c0 = initializer(batch_size * self.num_particles, self.hidden_dim)
        p0 = torch.ones(batch_size * self.num_particles, 1) * np.log(1 / self.num_particles)
        hidden = (h0, c0, p0)

        return hidden
        
    def forward(self, prev_window):
        """
        Implements hidden state update given a particular batch of inputs with some given sequence length.

        Returns estimates over the esquence length.
        """
        
        embedding = torch.relu(self.act_embedding(prev_window))

        # repeat the input if using the PF-RNN
        embedding = embedding.repeat(self.num_particles, 1, 1)
        seq_len = embedding.size(1)
        hidden = self.init_hidden(self.args.batch_size)

        hidden_states = []
        probs = []

        for step in range(seq_len):
            # print(embedding[:, step, :].shape)
            hidden = self.rnn(embedding[:, step, :], hidden)
            hidden_states.append(hidden[0])
            probs.append(hidden[-1])

        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = self.hnn_dropout(hidden_states)

        probs = torch.stack(probs, dim=0)
        prob_reshape = probs.view([seq_len, self.num_particles, -1, 1])
        out_reshape = hidden_states.view([seq_len, self.num_particles, -1, self.hidden_dim])
        y = out_reshape * torch.exp(prob_reshape)
        y = torch.sum(y, dim=1)
        y = self.hidden2label(y)
        pf_labels = self.hidden2label(hidden_states)

        y_out = torch.tanh(y)
        pf_out = torch.tanh(pf_labels) #TODO: investigate function of sigmoid, print y_out and y
        
        return y_out, pf_out

    def step(self, prev_window, gt_pos, args):
        """
        Implements one training step of the RNN, given a batched input, batched output, and parameters

        Returns loss tensor based on difference between prediction and input
        """
        
        pred, pf_out = self.forward(prev_window)
        pred = pred.squeeze(2)
        
        gt_normalized = gt_pos.transpose(0,1).contiguous()
        
        sl = pred.size(0)
        bpdecay_params = np.exp(args.bpdecay * np.arange(sl))
        bpdecay_params = bpdecay_params / np.sum(bpdecay_params)

        bpdecay_params = torch.FloatTensor(bpdecay_params)
        bpdecay_params = bpdecay_params.unsqueeze(0)
        bpdecay_params = bpdecay_params.unsqueeze(2)
        # pred = pred.transpose(0, 1).contiguous()

        l2_pred_loss = torch.nn.functional.mse_loss(pred, gt_normalized, reduction='none') * bpdecay_params
        l1_pred_loss = torch.nn.functional.l1_loss(pred, gt_normalized, reduction='none') * bpdecay_params

        l2_loss = torch.sum(l2_pred_loss)
        l1_loss = torch.mean(l1_pred_loss)

        pred_loss = args.l2_weight * l2_loss + args.l1_weight * l1_loss

        total_loss = pred_loss

        loss_last = torch.nn.functional.mse_loss(pred[:, -1], gt_normalized[:, -1])

        return total_loss, loss_last, pred

class TrackingDataset(Dataset):
    """
    Implements simple wrapper on PyTorch Datset for our exact dataset
    """
    def __init__(self, data, output):
        self.data = data
        self.outputs = output
        self.seq_len = len(self.data[0])
        self.seq_num = len(self.data)

        self.samp_seq_len = None

    def __len__(self):
        return self.seq_num

    def set_samp_seq_len(self, seq_len):
        self.samp_seq_len = seq_len

    def __getitem__(self, index):
        seq_idx = index % self.seq_num

        output = self.outputs[seq_idx]
        window = self.data[seq_idx]

        output = torch.FloatTensor(output)
        window = torch.FloatTensor(window)

        if self.samp_seq_len is not None and self.samp_seq_len != self.seq_len:
            start = np.random.randint(0, self.seq_len - self.samp_seq_len + 1)
            output = output[start:start + self.samp_seq_len]
            window = window[start:start + self.samp_seq_len]

        return (output, window)
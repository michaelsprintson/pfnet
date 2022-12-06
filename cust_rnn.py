import torch.nn as nn
import torch
from pfrnns import PFLSTMCell
import numpy as np
from torch.utils.data.dataset import Dataset

class SimpleRNN(nn.Module):
    """
    Implements RNN structure for given batch and arguments
    """
    def __init__(self, args):
        """
        Initialize RNN with input arguments and start the LSTM cell
        """
        super(SimpleRNN, self).__init__()
        self.batch_size = args.batch_size
        self.window_length = args.window_length
        self.hidden_dim = args.h
        self.act_emb = args.emb_act
        self.dropout_rate = args.dropout
        self.num_obs = args.obs_num
 
        self.rnn = nn.LSTM(self.act_emb, self.hidden_dim, batch_first = True) #might possibly have to do batch first True

        self.hidden2label = nn.Linear(self.hidden_dim, 1)

        self.act_embedding = nn.Linear(args.window_length*5, self.act_emb)
        self.hnn_dropout = nn.Dropout(self.dropout_rate)

        self.initialize = 'rand'
        self.args = args
        self.bp_length = args.bp_length

    def init_hidden(self, batch_size):
        """
        Initialize hidden states for the LSTM
        """
        initializer = torch.rand if self.initialize == 'rand' else torch.zeros

        h0 = initializer(1, batch_size, self.hidden_dim)
        c0 = initializer(1, batch_size, self.hidden_dim)
        hidden = (h0, c0)

        return hidden
        
    def forward(self, prev_window):
        """
        Implements hidden state update given a particular batch of inputs with some given sequence length.

        Returns estimates over the esquence length.
        """
        embedding = torch.relu(self.act_embedding(prev_window))

        # repeat the input if using the PF-RNN
        seq_len = embedding.size(1)
        hidden = self.init_hidden(self.args.batch_size)

        hidden_states = []

        for step in range(seq_len):
            e = embedding[:, step, :].unsqueeze(1)

            out, hidden= self.rnn(e, hidden)
            hidden_states.append(hidden[0])

        hidden_states = torch.stack(hidden_states, dim=0).squeeze(1)
        hidden_states = self.hnn_dropout(hidden_states)
        y = self.hidden2label(hidden_states)

        y_out = torch.sigmoid(y)
        
        return y_out

    def step(self, prev_window, gt_pos, args):
        """
        Implements one training step of the RNN, given a batched input, batched output, and parameters

        Returns loss tensor based on difference between prediction and input
        """

        # pred, particle_pred = self.forward(prev_window)
        pred = self.forward(prev_window)
        pred = pred.squeeze(2)
        # particle_pred = particle_pred.squeeze(2)
        
        #TODO:FIND LEN OF ABOVE 2
        gt_normalized = gt_pos.transpose(0,1).contiguous()
        # print(pred.shape)
        # print(gt_normalized.shape)

        batch_size = pred.size(1)
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

        return total_loss, loss_last, None

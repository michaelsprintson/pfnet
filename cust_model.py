import torch.nn as nn
import torch
from pfrnns import PFLSTMCell
import numpy as np
from torch.utils.data.dataset import Dataset

class Tracker(nn.Module):

    def __init__(self, args):
        super(Tracker, self).__init__()
        self.num_particles = args.num_particles
        self.batch_size = args.batch_size
        self.window_length = args.window_length
        self.hidden_dim = args.h
        self.obs_emb = args.emb_obs
        self.act_emb = args.emb_act
        self.dropout_rate = args.dropout
        total_emb = self.obs_emb + self.act_emb
        self.num_obs = args.obs_num

        self.rnn = PFLSTMCell(self.num_particles, total_emb,
            self.hidden_dim, 32, 32, args.resamp_alpha)

        self.hidden2label = nn.Linear(self.hidden_dim, 1)

        self.act_embedding = nn.Linear(args.window_length, self.act_emb)
        self.hnn_dropout = nn.Dropout(self.dropout_rate)

        self.initialize = 'rand'
        self.args = args
        self.bp_length = args.bp_length

    def init_hidden(self, batch_size):
        initializer = torch.rand if self.initialize == 'rand' else torch.zeros

        h0 = initializer(batch_size * self.num_particles, self.hidden_dim)
        c0 = initializer(batch_size * self.num_particles, self.hidden_dim)
        p0 = torch.ones(batch_size * self.num_particles, 1) * np.log(1 / self.num_particles)
        hidden = (h0, c0, p0)

        return hidden

    def detach_hidden(self, hidden):
        return tuple([h.detach() for h in hidden])
        
    def forward(self, prev_window):
        
        embedding = torch.relu(self.act_embedding(prev_window))

        # repeat the input if using the PF-RNN
        embedding = embedding.repeat(self.num_particles, 1, 1)
        seq_len = embedding.size(1)
        hidden = self.init_hidden(self.args.batch_size)

        hidden_states = []
        probs = []

        for step in range(seq_len):
            hidden = self.rnn(embedding[:, step, :], hidden)
            hidden_states.append(hidden[0])
            probs.append(hidden[-1])

            # if step % self.bp_length == 0:
            #     hidden = self.detach_hidden(hidden)

        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = self.hnn_dropout(hidden_states)

        probs = torch.stack(probs, dim=0)
        prob_reshape = probs.view([seq_len, self.num_particles, -1, 1])
        out_reshape = hidden_states.view([seq_len, self.num_particles, -1, self.hidden_dim])
        y = out_reshape * torch.exp(prob_reshape)
        y = torch.sum(y, dim=1)
        y = self.hidden2label(y)
        pf_labels = self.hidden2label(hidden_states)

        y_out = torch.sigmoid(y)
        pf_out = torch.sigmoid(pf_labels)
        
        return y_out, pf_out

    def step(self, prev_window, gt_pos, args):

        pred, particle_pred = self.forward(prev_window)
        #TODO:FIND LEN OF ABOVE 2
        gt_normalized = gt_pos

        batch_size = pred.size(1)
        sl = pred.size(0)
        bpdecay_params = np.exp(args.bpdecay * np.arange(sl))
        bpdecay_params = bpdecay_params / np.sum(bpdecay_params)

        bpdecay_params = bpdecay_params.unsqueeze(0)
        bpdecay_params = bpdecay_params.unsqueeze(2)
        pred = pred.transpose(0, 1).contiguous()

        l2_pred_loss = torch.nn.functional.mse_loss(pred, gt_normalized, reduction='none') * bpdecay_params
        l1_pred_loss = torch.nn.functional.l1_loss(pred, gt_normalized, reduction='none') * bpdecay_params

        l2_loss = torch.sum(l2_pred_loss)
        l1_loss = torch.mean(l1_pred_loss)

        pred_loss = args.l2_weight * l2_loss + args.l1_weight * l1_loss

        total_loss = pred_loss

        particle_pred = particle_pred.transpose(0, 1).contiguous()
        particle_gt = gt_normalized.repeat(self.num_particles, 1, 1)
        l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
        l1_particle_loss = torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params

        # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
        # other more complicated distributions could be used to improve the performance
        y_prob_l2 = torch.exp(-l2_particle_loss).view(self.num_particles, -1, sl)
        l2_particle_loss = - y_prob_l2.mean(dim=0).log()

        y_prob_l1 = torch.exp(-l1_particle_loss).view(self.num_particles, -1, sl)
        l1_particle_loss = - y_prob_l1.mean(dim=0).log()

        l2_particle_loss = torch.mean(l2_particle_loss)
        l1_particle_loss = torch.mean(l1_particle_loss) #TODO: why mean mean instead of sum mean

        belief_loss = args.l2_weight * l2_particle_loss + args.l1_weight * l1_particle_loss
        total_loss = total_loss + args.elbo_weight * belief_loss

        loss_last = torch.nn.functional.mse_loss(pred[:, -1], gt_pos[:, -1])

        particle_pred = particle_pred.view(self.num_particles, batch_size, sl)

        return total_loss, loss_last, particle_pred

class TrackingDataset(Dataset):
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
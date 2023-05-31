# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


# input_dim=params.n_pid,output_dim=params.n_question
class GRUDINA(nn.Module):
    def __init__(self, input_dim, dropout, hidden_dim, num_layers, output_dim, l2=1e-5,
                 cell_type="gru", separate_qa=False):
        super(GRUDINA, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.l2 = l2
        self.cell_type = cell_type
        self.separate_qa = separate_qa
        self.rnn = None
        if cell_type.lower() == "lstm":
            pass
        elif cell_type.lower() == "rnn":
            pass
        elif cell_type.lower() == "gru":
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers,
                              batch_first=True, dropout=self.dropout,
                              )
        if self.rnn is None:
            raise ValueError("cell type only support lstm, rnn or gru type.")
        self.pid_embedding = nn.Embedding(self.input_dim, self.hidden_dim)
        self.q_embedding = nn.Embedding(self.output_dim, self.hidden_dim)

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, q_data, qa_data, matrix, slip, guess, target, pid_data=None):
        q_embed_data = self.pid_embedding(q_data)

        if self.separate_qa:
            qa_embed_data = self.pid_embedding(qa_data)
            input_embed_data = torch.cat([q_embed_data, qa_embed_data], dim=-1)
        else:
            input_embed_data = q_embed_data

        gru_out, _ = self.rnn(input_embed_data)

        slip_probs = slip.unsqueeze(0)
        guess_probs = guess.unsqueeze(0)
        predicted_probs = gru_out.sigmoid()
        combined_probs = slip_probs * guess_probs + (1 - slip_probs) * (1 - guess_probs)
        dina_output = predicted_probs * combined_probs + (1 - predicted_probs) * (1 - combined_probs)
        output = self.fc(dina_output)

        return output

    def estimate_dina_params(self, qa_data, matrix):
        pass

# -*- coding:utf-8 -*-
import numpy as np
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
        # 题目嵌入，知识点嵌入
        self.pid_embedding = nn.Embedding(self.input_dim + 1, self.hidden_dim)
        self.q_embedding = nn.Embedding(self.output_dim + 1, self.hidden_dim)
        self.qa_embedding = nn.Embedding(2 * self.output_dim + 1, hidden_dim)

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.slip = nn.Parameter(torch.tensor([0.2]))  # 失误率
        self.guess = nn.Parameter(torch.tensor([0.2]))  # 猜测率

    def forward(self, q_data, qa_data, matrix, target, pid_data=None):
        # Batch First
        q_embed_data = self.q_embedding(q_data)  # 知识点嵌入
        qa_embed_data = self.qa_embedding(qa_data)  # 回答序列嵌入
        if pid_data is not None:  # 题目嵌入
            pid_embed_data = self.pid_embedding(pid_data)
        else:
            pid_embed_data = None

        # concatenate embedding
        if self.separate_qa:  # TRUE，把题目和回答序列分开处理
            input_embed_data = qa_embed_data
        else:  # FALSE，把题目和回答序列拼接在一起
            input_embed_data = torch.cat([q_embed_data, qa_embed_data, pid_embed_data], dim=-1)

        # 计算学生对知识点的掌握情况
        gru_out, _ = self.rnn(input_embed_data)
        gru_out=self.fc(gru_out)
        # 进行矩阵乘法
        # 学生的潜在作答情况，值大于0.6就设置为1，否则不做调整
        p_c_relations = matrix
        potential_responses = torch.bmm(gru_out, p_c_relations.t())
        potential_responses = np.where(potential_responses > 0, 1, potential_responses)

        # 计算学生答对题目的概率
        slip_prob = self.sigmoid(self.slip)
        guess_prob = self.sigmoid(self.guess)
        response_probs = (1 - slip_prob) * potential_responses + guess_prob * (1 - potential_responses)
        output = self.fc(response_probs)
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output.reshape(-1))  # logit
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        # l2正则化
        # l2_loss = torch.tensor(0.)
        # for param in self.parameters():
        #     l2_loss += torch.norm(param, p=2)  # Calculate L2 norm for each parameter
        # output = output.mean() + self.l2 * l2_loss

        # slip_probs = slip.unsqueeze(0)
        # guess_probs = guess.unsqueeze(0)
        # predicted_probs = gru_out.sigmoid()
        # combined_probs = slip_probs * guess_probs + (1 - slip_probs) * (1 - guess_probs)
        # dina_output = predicted_probs * combined_probs + (1 - predicted_probs) * (1 - combined_probs)
        # output = self.fc(dina_output)
        return output.sum(), m(preds), mask.sum()

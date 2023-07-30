# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device= [0, 1, 2]

class GRUDINA(nn.Module):
    # q_embed_dim=50 qa_embed_dim = 256 hidden_dim = 256 output_dim = 110
    def __init__(self, n_pid, n_question, q_embed_dim, qa_embed_dim, dropout, hidden_dim, output_dim, num_layers,
                 l2=1e-5, cell_type="gru", final_fc_dim=512, separate_qa=False):
        super(GRUDINA, self).__init__()
        self.n_pid = n_pid
        self.n_question = n_question
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.l2 = l2
        self.cell_type = cell_type
        self.separate_qa = separate_qa
        self.rnn = None

        # 题目嵌入，知识点嵌入，回答嵌入
        self.pid_embedding = nn.Embedding(self.n_pid + 1, self.q_embed_dim)
        self.q_embedding = nn.Embedding(self.n_question + 1, self.q_embed_dim)
        self.qa_embedding = nn.Embedding(2, self.q_embed_dim)
        if self.n_pid > 0:  # 难度嵌入
            self.difficult_parm = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, self.q_embed_dim)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, self.q_embed_dim)

        if cell_type.lower() == "lstm":
            self.rnn = nn.LSTM(self.pid_embedding + qa_embed_dim, self.hidden_dim, self.num_layers,
                               batch_first=True, dropout=self.dropout,
                               )
        elif cell_type.lower() == "rnn":
            self.rnn = nn.RNN(self.pid_embedding + qa_embed_dim, self.hidden_dim, self.num_layers,
                              batch_first=True, dropout=self.dropout,
                              )
        elif cell_type.lower() == "gru":
            self.rnn = nn.GRU(self.q_embed_dim + self.q_embed_dim, self.hidden_dim, self.num_layers,
                              batch_first=True, dropout=self.dropout,
                              )
        if self.rnn is None:
            raise ValueError("cell type only support lstm, rnn or gru type.")

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, q_data, qa_data, matrix, target, pid_data=None):
        BS = q_data.size(0)  # batch_size
        seqencelen = q_data.size(1)  # 序列长度
        # Batch First
        q_embed_data = self.q_embedding(q_data)  # c_ct,知识点嵌入,[BS, seqlen,  q_embed_dim]
        # 回答序列嵌入（概念响应嵌入）[BS, seqlen,  q_embed_dim],  g_rt + c_ct = e_(ct,rt)
        qa_data = torch.div(qa_data - q_data, self.n_question, rounding_mode='floor')  # 原始响应序列
        qa_embed_data = self.qa_embedding(qa_data) + q_embed_data
        # 题目嵌入
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct,总结了涵盖这个概念的问题的难度变化,[BS, seqlen, q_embed_dim]
            pid_embed_data = self.difficult_parm(pid_data)  # μ_qt标量，难度参数，控制该问题与概念的偏离程度
            # pid_embed_data = self.difficult_parm(pid_data) if pid_data is not None else pid_embed_data = None
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data  # e_ct = c_ct +μ_qt * d_ct ,[BS, seqlen, q_embed_dim]
            qa_embed_diff_data = self.qa_embed_diff(qa_data)  # variation vectors,[BS, seqlen,  q_embed_dim]
            qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data  # = (g_rt + c_ct) + μ_qt * ,[BS, seqlen, q_embed_dim]
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        # concatenate embedding
        if self.separate_qa:  # TRUE，把题目和回答序列分开处理
            input_embed_data = qa_embed_data
        else:  # FALSE，把题目和回答序列拼接在一起
            input_embed_data = torch.cat([qa_embed_data, q_embed_data], dim=-1)

        gru_out, _ = self.rnn(input_embed_data)
        gru_out = self.fc(gru_out)

        # 计算学生对知识点的掌握情况,只要最后一个时间步的输出
        # 进行矩阵乘法,gru_out:[BS, seqlen, output_dim], p_c_relations:[n_question, n_question]
        p_c_relations = matrix
        result_master = []
        guess = []
        slip = []
        pred = []
        for bs in range(BS):  # self.batch_size,,,,,能否改进
            # 取出当前学生的 gru_out
            gru_out_student = gru_out[bs]  # 形状为 [seqlen, output_dim]
            # 进行矩阵点乘运算，注意需要将 gru_out_student 调整为正确的形状
            got_student = torch.matmul(gru_out_student, p_c_relations.t())  # 形状为 [seqlen, n_question]
            # 如果每一个时间步的每一个知识点的掌握情况元素值大于等于0.4，则认为该知识点掌握，值重置为1，否则认为该知识点没有掌握，值重置为0
            got_student[got_student >= 0.4] = 1
            result_master.append(got_student)

            # 初始化猜测率和失误率为0，取出当前学生的答案序列、知识点序列
            q_data_one = q_data[bs]  # 形状为 [seqlen]
            qa_data_one = qa_data[bs]  # 形状为 [seqlen]
            guessing_rate_one = torch.zeros(seqencelen, self.output_dim)
            slipping_rate_one = torch.zeros(seqencelen, self.output_dim)
            result_one = []
            current_master = got_student  # 学生的概念掌握情况
            for t in range(seqencelen):
                if t >= 1:
                    guessing_rate_one[t] = guessing_rate_one[t - 1]
                    slipping_rate_one[t] = slipping_rate_one[t - 1]
                qa_t = qa_data_one[:t + 1]  # 当前学生的答案序列
                q_t = q_data_one[:t + 1]  # 当前学生的知识点序列
                current_q = q_t[-1]  # 当前回答的知识点
                answered_all = (q_t == current_q).sum()  # 在t时间步前，回答过该知识点的次数
                master_correct = 0
                master_incorrect = 0
                no_master_correct = 0
                no_master_incorrect = 0
                for k in range(t):
                    if (q_t[k] == current_q) & (qa_t[k] == 1) & (current_master[k][q_t[k] - 1] == 1):
                        master_correct += 1
                    if (q_t[k] == current_q) & (qa_t[k] == 0) & (current_master[k][q_t[k] - 1] == 1):
                        master_incorrect += 1
                    if (q_t[k] == current_q) & (qa_t[k] == 0) & (current_master[k][q_t[k] - 1] == 0):
                        no_master_correct += 1
                    if (q_t[k] == current_q) & (qa_t[k] == 1) & (current_master[k][q_t[k] - 1] == 0):
                        no_master_incorrect += 1
                # 算当前回答的知识点的猜测率和失误率
                if (current_master[t][current_q - 1] == 1):
                    if (qa_t[t] == 1):
                        guessing_rate_one[t][current_q - 1] = master_correct / answered_all
                    else:
                        slipping_rate_one[t][current_q - 1] = master_incorrect / answered_all
                else:
                    if (qa_t[t] == 0):
                        guessing_rate_one[t][current_q - 1] = 1 - (no_master_correct / answered_all)
                    else:
                        guessing_rate_one[t][current_q - 1] = no_master_correct / answered_all
                # 输出当前时间步所回答的该知识点的预测结果
                # prediction=(1 - slip) * (concept_master * guess + (1 - slip) * (1 - concept_master))
                result_one.append((1 - slipping_rate_one[t][current_q - 1]) * (
                        current_master[t][current_q - 1] * guessing_rate_one[t][current_q - 1] + (
                        1 - slipping_rate_one[t][current_q - 1]) * (1 - current_master[t][current_q - 1])))

            pred.append(result_one)
            guess.append(guessing_rate_one)
            slip.append(slipping_rate_one)
        # # 将结果堆叠为一个张量
        # concept_master = torch.stack(result_master, dim=0)  # 形状为 [BS, seqlen, n_question]
        # guess = torch.stack(guess, dim=0)  # 形状为 [BS, seqlen, n_question]
        # slip = torch.stack(slip, dim=0)  # 形状为 [BS, seqlen, n_question]
        # # 只要seqlen中最后一个时间步的概念掌握情况，[BS, -1, n_question]
        # concept_master = concept_master[:, -1, :]  # concept_master = concept_master[:, -1, :].unsqueeze(1)
        # guess = guess[:, -1, :]  # guess = guess[:, -1, :].unsqueeze(1)
        # slip = slip[:, -1, :]  # slip = slip[:, -1, :].unsqueeze(1)

        # 学生的潜在作答情况，值大于0.6就设置为1，否则不做调整
        # potential_responses = np.where(potential_responses > 0, 1, potential_responses)
        pred = torch.stack([torch.tensor(p) for p in pred], dim=0)
        preds = pred.reshape(-1)
        labels = target.reshape(-1)
        m = nn.Sigmoid() # 创建一个 Sigmoid 模块的实例，然后可以使用该模块对张量进行 sigmoid 变换
        mask = labels > -0.9
        masked_labels = labels[mask].float().to(device[1])
        masked_preds = preds[mask].to(device[1])
        #MSELoss
        criterion = nn.MSELoss(reduction='none')
        criterion.to(device[1])
        output = criterion(masked_preds, masked_labels)

        # combined_probs = slip_probs * guess_probs + (1 - slip_probs) * (1 - guess_probs)
        # dina_output = predicted_probs * combined_probs + (1 - predicted_probs) * (1 - combined_probs)
        #  m(preds)对preds每个元素sigmoid变换，将模型的预测结果压缩到 [0, 1] 的范围
        return output.sum() + c_reg_loss, m(preds), mask.sum()

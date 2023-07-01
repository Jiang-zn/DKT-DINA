import torch
import torch.nn as nn
from torch.autograd import Variable


# DKT模型需要把学生答题记录转换成学生答题特征矩阵，这个特征最后会作为输出，输入到IRT模型中
class DKTIRT(nn.Module):
    def __init__(self, n_question, q_embed_dim, qa_embed_dim, hidden_dim, layer_dim, output_dim, dropout, device="cpu",
                 cell_type="lstm"):
        """ The first deep knowledge tracing network architecture.

        :param embed_dim: int, the embedding dim for each skill.
        :param input_dim: int, the number of skill(question) * 2.
        :param hidden_dim: int, the number of hidden state dim.
        :param layer_num: int, the layer number of the sequence number.
        :param output_dim: int, the number of skill(question).
        :param device: str, 'cpu' or 'cuda:0', the default value is 'cpu'.
        :param cell_type: str, the sequence model type, it should be 'lstm', 'rnn' or 'gru'.
        """
        super(DKTIRT, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.skill_embedding = nn.Embedding(self.input_dim, self.embed_dim, padding_idx=self.input_dim - 1)
        # 暂时只用LSTM
        self.cell_type = cell_type
        self.rnn = None
        if cell_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                self.embed_dim,
                self.hidden_dim,
                self.layer_num,
                batch_first=True,
                dropout=self.dropout,
            )
        elif cell_type.lower() == "rnn":
            self.rnn = nn.RNN(
                self.embed_dim,
                self.hidden_dim,
                self.layer_num,
                batch_first=True,
                dropout=self.dropout,
            )
        elif cell_type.lower() == "gru":
            self.rnn = nn.GRU(
                self.embed_dim,
                self.hidden_dim,
                self.layer_num,
                batch_first=True,
                dropout=self.dropout,
            )
        # output layer to predict student mastery of each skill
        self.fc = nn.Linear(hidden_dim, output_dim)
        if self.rnn is None:
            raise ValueError("The cell type should be 'lstm', 'rnn' or 'gru'.")

    def forward(self, qa):
        qa = self.skill_embedding(qa)
        h0 = torch.zeros(self.num_layers, qa.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, qa.size(0), self.hidden_dim)
        # h0 = torch.zeros((self.num_layers, qa.size(0), self.hidden_dim), device=self.device)
        # c0 = torch.zeros((self.num_layers, qa.size(0), self.hidden_dim), device=self.device)
        # pass the input sequence through the LSTM layer
        state, state_out = self.rnn(qa, (h0, c0))
        logits = torch.sigmoid(self.fc(state))
        return logits

# IRT模型需要有三个主要参数：包括题目的难度、学生的能力和题目的区分度，
# 确定了这三个参数的值，才能更好的把学生答题特征和题目-知识点关联矩阵输入到IRT模型中
# 运行模型并得到学生能力水平预测结果
# x: 学生答题特征矩阵,形状为(batch_size, num_skills)
# q: 学生答题序列中的问题知识点编号向量,形状为(batch_size,)
# c: 学生答题序列中的题目-知识点关联矩阵,形状为(num_skills,num_questions)

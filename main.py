import sys
import os
import os.path
import glob
import logging
import argparse
import numpy as np
import torch
from DKTIRT import *
from load_data import *
from eval import *
from p_c_matrix import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# assert torch.cuda.is_available(), "No Cuda available, AssertionError"

if __name__ == '__main__':
    # 参数设置 Parse Arguments Basic Parameters Common parameters Datasets and Model
    parser = argparse.ArgumentParser(description='test DKT-IRT')
    parser.add_argument('--train_set', type=int, default=1)
    parser.add_argument('--seed', type=int, default=224, help='默认随机种子')
    parser.add_argument('--max_iter', type=int, default=300, help='迭代次数')
    parser.add_argument('--optim', type=str, default='adam', help='默认优化器')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--maxgradnorm', type=float, default=-1, help='最大化梯度范数')
    parser.add_argument('--final_fc_dim', type=int, default=512, help='fc层维度')
    parser.add_argument('--model', type=str, default='DKT-IRT_pid',
                        help="combination of akt/dkvmn/dkt, pid/cid separated by underscore '_'. For example tf_pid")
    parser.add_argument('--dataset', type=str, default="assist2009_pid")

    # 用于IRT模型，难度，区分度，暂时不知道怎么用
    parser.add_argument('--num_difficulties', type=int, default=100, help='number of difficulties')
    parser.add_argument('--num_discriminabilities', type=int, default=100, help='number of discriminabilities')

    # DKT Specific Parameter
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lamda_r', type=float, default=0.1)
    parser.add_argument('--lamda_w1', type=float, default=0.1)
    parser.add_argument('--lamda_w2', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.5)

    # DKT-VMN Specific  Parameter
    parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
    parser.add_argument('--qa_embed_dim', type=int, default=256, help='answer and question embedding dimensions')
    parser.add_argument('--memory_size', type=int, default=50, help='memory size')
    parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')

    parser.add_argument('--hidden_size', type=int, default=100, help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=1, help='隐藏层层数')
    parser.add_argument('--bidirectional', type=bool, default=False, help='是否使用双向RNN模型')

    params = parser.parse_args()

    dataset = params.dataset
    if dataset in {"assist2009_pid"}:
        params.batch_size = 24
        params.n_question = 110
        params.seqlen = 200
        params.n_pid = 16891
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
    if dataset in {"assist2017_pid"}:
        params.batch_size = 24
        params.n_question = 102
        params.seqlen = 200
        params.n_pid = 3162
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
    if dataset in {"assist2015"}:
        params.batch_size = 24
        params.n_question = 100
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
    if dataset in {"statics"}:
        params.batch_size = 24
        params.n_question = 1223
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
    params.save = params.data_name
    params.load = params.data_name

    # setup, choose the load_data function
    if "pid" not in params.data_name:
        final_data = DATA(n_question=params.n_question, seqlen=params.seqlen, separate_char=',')
    else:
        final_data = PID_DATA(n_question=params.n_question, seqlen=params.seqlen, separate_char=',')
    seedNum = params.seed
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    model_info = get_model_info(params)
    print('model_info', model_info)

    # 打印参数表
    # d = vars(params)
    # for key in d:
    #     print('\t', key, '\t', d[key])
    file_name = ''
    for item_ in model_info:
        file_name = file_name + item_[0] + str(item_[1])
    # print(file_name)

    # for filenums in range(1, 6):
    # params.train_set = filenums
    train_data_path = params.data_dir + "/" + params.data_name + "_train" + str(params.train_set) + ".csv"
    valid_data_path = params.data_dir + "/" + params.data_name + "_valid" + str(params.train_set) + ".csv"
    train_q_data, train_qa_data, train_pid, train_matrix = final_data.load_data(train_data_path)
    valid_q_data, valid_qa_data, valid_pid, valid_matrix = final_data.load_data(valid_data_path)
    print("\n")
    # print('now training file{}'.format(filenums))
    print("train_q_data.shape", train_q_data.shape)
    print("train_qa_data.shape", train_qa_data.shape)
    print("valid_q_data.shape", valid_q_data.shape)
    print("valid_qa_data.shape", valid_qa_data.shape)
    print("\n")

    # Train and get the best episode
    # best_epoch = train_one_dataset(params, train_q_data, train_qa_data, train_pid, train_matrix, valid_q_data, valid_qa_data, valid_pid, valid_matrix, file_name)
    # test_data_path = params.data_dir + "/" + params.data_name + "_test" + str(params.train_set) + ".csv"
    # test_q_data, test_qa_data, test_pid = final_data.load_data(test_data_path)
    # test_one_dataset(params, file_name, test_q_data, test_qa_data, test_index, best_epoch)

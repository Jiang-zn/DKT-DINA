import sys
import os
import os.path
import glob
import logging
import argparse
import numpy as np
import torch
from model import *
from load_data import *
from eval import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# assert torch.cuda.is_available(), "No Cuda available, AssertionError"

if __name__ == '__main__':
    # 参数设置 Parse Arguments Basic Parameters Common parameters Datasets and Model
    parser = argparse.ArgumentParser(description='test DKT-DINA')
    parser.add_argument('--train_set', type=int, default=1)
    parser.add_argument('--seed', type=int, default=224, help='默认随机种子')
    parser.add_argument('--max_iter', type=int, default=1, help='迭代次数')
    parser.add_argument('--optim', type=str, default='adam', help='默认优化器')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default='GRU-DINA_pid',
                        help="combination of akt/dkvmn/dkt, pid/cid separated by underscore '_'. For example tf_pid")
    parser.add_argument('--dataset', type=str, default="assist2009_pid")

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--qa_embed_dim', type=int, default=256, help='answer and question embedding dimensions')
    parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
    parser.add_argument('--num_layers', type=int, default=1, help='隐藏层层数')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=110, help='output dimensions')
    parser.add_argument('--bidirectional', type=bool, default=False, help='是否使用双向RNN模型')
    # AKT-R Specific Parameter
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty for difficulty')
    # parser.add_argument('--maxgradnorm', type=float, default=-1, help='最大化梯度范数')
    parser.add_argument('--final_fc_dim', type=int, default=512, help='fc层维度')
    # parser.add_argument('--num_difficulties', type=int, default=100, help='number of difficulties')
    # parser.add_argument('--num_discriminabilities', type=int, default=100, help='number of discriminabilities')

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
        dat = DATA(n_question=params.n_question, seqlen=params.seqlen, separate_char=',')
    else:
        dat = PID_DATA(n_question=params.n_question, n_pid=params.n_pid, seqlen=params.seqlen, separate_char=',')
    seedNum = params.seed
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    file_name_identifier = get_model_info(params)

    # Train- Test,参数打印
    d = vars(params)
    for key in d:
        print('\t', key, '\t', d[key])
    file_name = ''
    for item_ in file_name_identifier:
        file_name = file_name + item_[0] + str(item_[1])
    # print(file_name)

    # 构建题目-知识点关联矩阵，并写进csv文件中
    # paths = []
    # for file_num in range(1, 6):
    #     train_data_path = params.data_dir + "/" + params.data_name + "_train" + str(file_num) + ".csv"
    #     valid_data_path = params.data_dir + "/" + params.data_name + "_valid" + str(file_num) + ".csv"
    #     test_data_path = params.data_dir + "/" + params.data_name + "_test" + str(file_num) + ".csv"
    #     paths.append(train_data_path)
    #     paths.append(valid_data_path)
    #     paths.append(test_data_path)
    # init_matrix = np.zeros((params.n_pid + 1, params.n_question + 1))
    # p_c_matrix = dat.build_p_c_matrix(paths, init_matrix)
    # np.savetxt('p_c_matrix.csv', p_c_matrix, delimiter=',', fmt='%.1f')
    # 读取这个文件中的数据
    # 把题目-知识点关联矩阵读取出来
    p_c_matrix = np.loadtxt('p_c_matrix.csv', delimiter=',')  # shape(16892, 111),但实际上是16891个题目，110个知识点

    train_data_path = params.data_dir + "/" + params.data_name + "_train1" + ".csv"
    # valid_data_path = params.data_dir + "/" + params.data_name + "_valid1" + ".csv"
    train_q_data, train_qa_data, train_pid= dat.load_data(train_data_path)

    # valid_q_data, valid_qa_data, valid_pid = dat.load_data(valid_data_path)
    # print(train_pid[0])
    # print(train_q_data[0])
    # print(train_qa_data[0])
    # print(np.floor((train_qa_data[0] - 1) / 110))

    # print("\n")
    # print("train_q_data.shape", train_q_data.shape)
    # print("train_qa_data.shape", train_qa_data.shape)
    # print("valid_q_data.shape", valid_q_data.shape)  # (1566, 200)
    # print("valid_qa_data.shape", valid_qa_data.shape)  # (1566, 200)
    # print("\n")

    # best_epoch = train_one_dataset(params, file_name, train_q_data, train_qa_data, train_pid, valid_q_data,
    #                                valid_qa_data, valid_pid)
    # test_data_path = params.data_dir + "/" + params.data_name + "_test" + str(filenums) + ".csv"
    # test_q_data, test_qa_data, test_pid = dat.load_data(test_data_path)
    # test_one_dataset(params, file_name, test_q_data, test_qa_data, test_pid, best_epoch)

import sys
import os
import os.path
import glob
import logging
import argparse
import time
import numpy as np
import torch
from load_data import *
from eval import *
from model import *
from utils import *
from grudina import GRUDINA
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = [0,1,2]


# assert torch.cuda.is_available(), "No Cuda available, AssertionError"


# 进行params.max_iter次迭代，每次迭代都会进行一次完整的训练和一次完整的验证，
# 记录每次迭代的训练和验证的loss、accuracy、auc，
# 在训练过程中定期保存最好的模型，并删除之前保存的模型，以避免存储大量模型文件
# 输出到底是哪一次epoch的模型最好
def train_one_dataset(params, file_name, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data,
                      valid_pid, matrix):
    # ================================== model initialization ==================================

    model = load_model(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-08)
    print("\n")

    # ================================== training ==================================
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0
    matrix = matrix
    for index in range(params.max_iter):
        # 设置每一次迭代的开始时间，最后获取结束时间，计算训练时间，输出格式为时分秒

        # Train Model
        train_loss, train_accuracy, train_auc = train(model, params, optimizer, train_q_data, train_qa_data, train_pid,
                                                      matrix, label='Train')
        # Validatation
        valid_loss, valid_accuracy, valid_auc = test(model, params, optimizer, valid_q_data, valid_qa_data, valid_pid,
                                                     matrix, label='Valid')

        print('epoch', idx + 1)
        print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
        print("valid_accuracy\t", valid_accuracy, "\ttrain_accuracy\t", train_accuracy)
        print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)
        print("time\t", time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)))
        try_makedirs('model')
        try_makedirs(os.path.join('model', params.model))
        try_makedirs(os.path.join('model', params.model, params.save))

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        # output the epoch with the best validation auc
        # 在训练过程中定期保存最好的模型，并删除之前保存的模型，以避免存储大量模型文件
        if valid_auc > best_valid_auc:
            path = os.path.join('model', params.model,
                                params.save, file_name) + '_*'
            for i in glob.glob(path):
                os.remove(i)
            best_valid_auc = valid_auc
            best_epoch = idx + 1
            torch.save({'epoch': idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        },
                       os.path.join('model', params.model, params.save,
                                    file_name) + '_' + str(idx + 1)
                       )
        # 提高模型的泛化能力，模型在过去40轮训练中没有表现更好的时候，就会停止训练并跳出循环。这是为了避免模型在过拟合的情况下继续训练
        if idx - best_epoch > 40:
            break

    try_makedirs('result')
    try_makedirs(os.path.join('result', params.model))
    try_makedirs(os.path.join('result', params.model, params.save))
    f_save_log = open(os.path.join(
        'result', params.model, params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    return best_epoch


# 加载之前训练好的模型权重，以便继续训练或进行预测，
# 在训练模型时，我们通常会将每个epoch训练的模型保存下来，以便之后进行评估或继续训练。
# 但是如果不加控制，这些模型文件可能会占用很大的磁盘空间，因此需要定期删除旧的模型文件
def test_one_dataset(params, file_name, test_q_data, test_qa_data, test_pid, best_epoch):
    print("\n\nStart testing ......................\n Best epoch:", best_epoch)
    model = load_model(params)

    checkpoint = torch.load(os.path.join(
        'model', params.model, params.save, file_name) + '_' + str(best_epoch))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_accuracy, test_auc = test(model, params, None, test_q_data, test_qa_data, test_pid, label='Test')
    print("\ntest_auc\t", test_auc)
    print("test_accuracy\t", test_accuracy)
    print("test_loss\t", test_loss)
    # Now Delete all the models
    path = os.path.join('model', params.model, params.save, file_name) + '_*'
    for i in glob.glob(path):
        os.remove(i)


if __name__ == '__main__':
    # 参数设置 Parse Arguments Basic Parameters Common parameters Datasets and Model
    parser = argparse.ArgumentParser(description='test Gru-Dina')
    parser.add_argument('--train_set', type=int, default=1)
    parser.add_argument('--seed', type=int, default=224, help='默认随机种子')
    parser.add_argument('--max_iter', type=int, default=1, help='迭代次数')
    parser.add_argument('--optim', type=str, default='adam', help='默认优化器')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default='GruDina_pid',
                        help="combination of akt/dkvmn/dkt, pid/cid separated by underscore '_'. For example tf_pid")
    parser.add_argument('--dataset', type=str, default="assist2009_pid")

    parser.add_argument('--dropout', type=float, default=0)
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
    prerequisite_relation_matrix = np.loadtxt('prerequisite-relation.csv', delimiter=',')  # shape(110, 110)

    train_data_path = params.data_dir + "/" + params.data_name + "_train1" + ".csv"
    valid_data_path = params.data_dir + "/" + params.data_name + "_valid1" + ".csv"
    train_q_data, train_qa_data, train_pid = dat.load_data(train_data_path)
    valid_q_data, valid_qa_data, valid_pid = dat.load_data(valid_data_path)

    print("\n")
    print("train_q_data.shape", train_q_data.shape)  # (2994, 200)
    print("train_qa_data.shape", train_qa_data.shape)  # (2994, 200)
    print("valid_q_data.shape", valid_q_data.shape)  # (974, 200)
    print("valid_qa_data.shape", valid_qa_data.shape)  # (974, 200)
    print("\n")
    # print(train_q_data[:6,:])
    # print(train_q_data[6:12,:])
    # print(train_q_data[12:18,:])
    # print(train_q_data[18:24,:])
    # print(train_q_data[24:30,:])
    # print(train_q_data[30:32,:])
    best_epoch = train_one_dataset(params, file_name, train_q_data, train_qa_data, train_pid, valid_q_data,
                                   valid_qa_data, valid_pid, prerequisite_relation_matrix)
    # test_data_path = params.data_dir + "/" + params.data_name + "_test" + str(filenums) + ".csv"
    # test_q_data, test_qa_data, test_pid = dat.load_data(test_data_path)
    # test_one_dataset(params, file_name, test_q_data, test_qa_data, test_pid, best_epoch)

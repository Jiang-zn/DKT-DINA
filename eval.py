import os.path

import numpy as np
import torch
from sklearn import metrics
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# transpose_data_model = {'DKT-DINA'}

# 二进制交叉熵损失函数
def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
           (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    if mod == 'avg':
        return np.average(loss) * (-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


# 计算AUC
def compute_auc(all_target, all_pred):
    # fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


# 计算准确率
def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params, optimizer, q_data, qa_data, pid_data, matrix, label):
    net.train()
    pid_flag, model_type = model_isPid(params.model)
    N = int(math.ceil(len(q_data) / params.batch_size))  # 一次epoch中的batch数，例如200/24
    q_data = q_data.T  # 原data是(2994,200) Shape: (200,2994)
    qa_data = qa_data.T  # Shape: (200,2994)
    # Shuffle the data，将题目数据和答案数据按列打乱顺序，增加数据随机性，减小模型过拟合的风险
    # shuffled_ind = np.arange(q_data.shape[1])
    # np.random.shuffle(shuffled_ind)
    # q_data = q_data[:, shuffled_ind]
    # qa_data = qa_data[:, shuffled_ind]
    if pid_flag:
        pid_data = pid_data.T
        # pid_data = pid_data[:, shuffled_ind]

    pred_list = []
    target_list = []
    element_count = 0
    true_el = 0
    for idx in range(N):
        optimizer.zero_grad()
        q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size]
        qa_one_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size]
        if pid_flag:
            pid_one_seq = pid_data[idx * params.batch_size:(idx + 1) * params.batch_size]

        # 这主要做的是转置操作将(200，bs) Shape (bs, 200)
        # 使用了batch_first=True
        # if model_type in transpose_data_model:
        #     input_q = np.transpose(q_one_seq[:, :])  # Shape (bs, seqlen)
        #     input_qa = np.transpose(qa_one_seq[:, :])  # Shape (bs, seqlen)
        #     target = np.transpose(qa_one_seq[:, :])
        #     if pid_flag:
        #         # Shape (seqlen, batch_size)
        #         input_pid = np.transpose(pid_one_seq[:, :])
        # else: # 这一步有待研究
        #     input_q = (q_one_seq[:, :])  # Shape (seqlen, batch_size)
        #     input_qa = (qa_one_seq[:, :])  # Shape (seqlen, batch_size)
        #     target = (qa_one_seq[:, :])
        #     if pid_flag:
        #         input_pid = (pid_one_seq[:, :])  # Shape (seqlen, batch_size)
        input_q = np.transpose(q_one_seq[:, :])
        input_qa = np.transpose(qa_one_seq[:, :])
        target = np.transpose(qa_one_seq[:, :])
        if pid_flag:
            input_pid = np.transpose(pid_one_seq[:, :])

        # 将target减1然后除以总知识数，这样可以将回答错误的标签映射到[0, 1]之间，
        # 回答正确的标签映射到[1, 2]之间，补的0映射到[-1, 0]之间
        # 1表示回答正确，0表示回答错误，-1表示补的0
        target = (target - 1) / params.n_question
        target_l = np.floor(target)
        el = np.sum(target_l >= -0.9)
        element_count += el  # target中答对的总数

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target).float().to(device)
        # if pid_flag:
        input_pid = torch.from_numpy(input_pid).long().to(device)
        # if pid_flag:
        loss, pred, true_ct = net(input_q, input_qa, matrix, target, input_pid)

        pred = pred.cpu().detach().numpy()
        loss.backward()
        true_el += true_ct.cpu().numpy()  # 预测答对的总数
        optimizer.step()
        # correct: 1.0; wrong 0.0; padding -1.0  将target_1重塑为一维数组
        target = target_l.reshape((-1,))
        # 将预测值和目标值中的有效数据（即非填充值）提取出来，并分别存储在pred_list和target_list中
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    # all_pred和all_target将包含所有数组在行方向上连接而成的结果
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc


def test(net, params, optimizer, q_data, qa_data, pid_data, label):
    # dataArray: [ array([[],[],..])] Shape: (2994, 200)
    pid_flag, model_type = model_isPid(params.model)
    net.eval()
    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    q_data = q_data.T  # Shape: (200,2994)
    qa_data = qa_data.T  # Shape: (200,2994)
    if pid_flag:
        pid_data = pid_data.T
    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []

    count = 0
    true_el = 0
    element_count = 0
    for idx in range(N):
        q_one_seq = q_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        qa_one_seq = qa_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        if pid_flag:
            pid_one_seq = pid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]

        # 这主要做的是转置操作将(200，bs) Shape (bs, 200)
        # 使用了batch_first=True
        # if model_type in transpose_data_model:
        #     input_q = np.transpose(q_one_seq[:, :])  # Shape (bs, seqlen)
        #     input_qa = np.transpose(qa_one_seq[:, :])  # Shape (bs, seqlen)
        #     target = np.transpose(qa_one_seq[:, :])
        #     if pid_flag:
        #         # Shape (seqlen, batch_size)
        #         input_pid = np.transpose(pid_one_seq[:, :])
        # else: # 这一步有待研究
        #     input_q = (q_one_seq[:, :])  # Shape (seqlen, batch_size)
        #     input_qa = (qa_one_seq[:, :])  # Shape (seqlen, batch_size)
        #     target = (qa_one_seq[:, :])
        #     if pid_flag:
        #         input_pid = (pid_one_seq[:, :])  # Shape (seqlen, batch_size)
        input_q = np.transpose(q_one_seq[:, :])
        input_qa = np.transpose(qa_one_seq[:, :])
        target = np.transpose(qa_one_seq[:, :])
        if pid_flag:
            input_pid = np.transpose(pid_one_seq[:, :])

        target = (target - 1) / params.n_question
        target_1 = np.floor(target)
        # target = np.random.randint(0,2, size = (target.shape[0],target.shape[1]))

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)

        with torch.no_grad():
            loss, pred, true_ct = net(input_q, input_qa, matrix, target, input_pid)

            # if pid_flag:
            #     loss, pred, ct = net(input_q, input_qa, target, input_pid)
            # else:
            #     loss, pred, ct = net(input_q, input_qa, target)
        pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
        true_el += ct.cpu().numpy()
        # target = target.cpu().numpy()
        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            count += real_batch_size
        else:
            count += params.batch_size

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        # print avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    assert count == seq_num, "Seq not matching"

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc


# 进行params.max_iter次迭代，每次迭代都会进行一次完整的训练和一次完整的验证，
# 记录每次迭代的训练和验证的loss、accuracy、auc，
# 在训练过程中定期保存最好的模型，并删除之前保存的模型，以避免存储大量模型文件
# 输出到底是哪一次epoch的模型最好
def train_one_dataset(params, file_name, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data,
                      valid_pid):
    # ================================== model initialization ==================================

    model = load_model(params)
    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-08)
    print("\n")

    # ================================== training ==================================
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for index in range(params.max_iter):
        # 设置每一次迭代的开始时间，最后获取结束时间，计算训练时间，输出格式为时分秒
        start_time = time.time()
        # Train Model
        train_loss, train_accuracy, train_auc = train(model, params, optimizer, train_q_data, train_qa_data, train_pid,
                                                      matrix, label='Train')
        # Validatation
        valid_loss, valid_accuracy, valid_auc = test(model, params, optimizer, valid_q_data, valid_qa_data, valid_pid,
                                                     label='Valid')

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


# 若文件夹不存在则创建文件夹
def open_dirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass


# 判断数据集是否有pid，返回是否有pid的布尔值和数据集名
def model_isPid(model_name):
    items = model_name.split('_')
    is_pid = True if "pid" in items else False
    return is_pid, items[0]


# 对所选用的模型进行参数设置，暂时只设置了DKT-IRT
# 这段代码是为了构造保存模型的文件名，以便于后续查找和使用不同的模型训练结果
def get_model_info(params):
    model_type = params.model.split('_')[0]
    # print("model_type", model_type)
    if model_type == 'DKT-IRT':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_ts', params.train_set],
                     ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2]]
    elif model_type == 'GRU-DINA':
        file_name = [['_b', params.batch_size], ['_lr', params.lr], ['_nl', params.num_layers],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_ts', params.train_set],
                     ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2]]
    elif model_type == 'DKT':
        pass
    elif model_type == 'AKT':
        pass
    return file_name


# 加载模型
def load_model(params):
    items = params.model.split('_')
    model_type = items[0]
    is_cid = items[1] == 'cid'
    if is_cid:
        params.n_pid = -1

    if model_type in {'DKT-IRT'}:
        model = DKTIRT(n_question=params.n_question, n_pid=params.n_pid, dropout=params.dropout).to(
            device)
    elif model_type in {'gd'}:
        model = GRUDINA(input_dim=params.n_pid, dropout=params.dropout, hidden_dim=params.hidden_dim,
                        num_layers=params.num_layers, output_dim=params.n_question,
                        l2=params.l2).to(device)
    else:
        model = None
    return model

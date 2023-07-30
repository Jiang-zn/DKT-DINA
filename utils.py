import os
import torch
import torch.nn as nn
from grudina import GRUDINA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = [0,1,2]


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
    if model_type == 'GruDina':
        file_name = [['_b', params.batch_size], ['_lr', params.lr], ['_nl', params.num_layers],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_ts', params.train_set],
                     ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2]]
    else:
        file_name = None
    return file_name


# 加载模型
def load_model(params):
    items = params.model.split('_')
    model_type = items[0]
    is_cid = items[1] == 'cid'
    if is_cid:
        params.n_pid = -1
    if model_type in {'DKT-IRT'}:
        model = None
    elif model_type in {'GruDina'}:
        model = GRUDINA(n_pid=params.n_pid, n_question=params.n_question, q_embed_dim=params.q_embed_dim,
                        qa_embed_dim=params.qa_embed_dim, dropout=params.dropout, hidden_dim=params.hidden_dim,
                        output_dim=params.output_dim, num_layers=params.num_layers, l2=params.l2)
        model = nn.DataParallel(model, device_ids=device)
        model = model.to(device[0])
    else:
        model = None
    return model

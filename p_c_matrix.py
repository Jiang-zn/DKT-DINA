import numpy as np
import math
from scipy import sparse


# 对每一个学生的数据构建题目-知识点关联矩阵
# 1.用enumerate遍历每一个学生的所有答题信息
# 2.对所回答的题目及对应回答的知识点进行标记，构建二维矩阵
# 3.将每个学生的题目-知识点关联矩阵存储到一个列表中
def build_p_c_matrix(path, n_question, n_skill):
    csv_data = open(path, 'r')

    # 读取第六行的数据
    p_c_matrix = []
    for lineID, line in enumerate(csv_data):
        line = line.strip()
        if lineID % 4 == 1:
            P = line.split(',')
            if len(P[len(P) - 1]) == 0:
                P = P[:-1]
        if lineID % 4 == 2:
            Q = line.split(',')
            if len(Q[len(Q) - 1]) == 0:
                Q = Q[:-1]
            # 构建一个二维矩阵，把P中的最大值是这个二维矩阵的行数，Q中的最大值作为这个二维矩阵的列数
            max_P = 0
            for i in range(len(P)):
                if int(P[i]) > max_P:
                    max_P = int(P[i])
            max_Q = 0
            for i in range(len(Q)):
                if int(Q[i]) > max_Q:
                    max_Q = int(Q[i])
            # 把矩阵的值都置为0
            p_c_matrix_one = np.zeros((max_P, max_Q))
            # print(p_c_matrix_one)
            # print(p_c_matrix_one.shape)
            # 遍历P和Q，把P中的值作为行，Q中的值作为列，把这个二维矩阵中的值置为1，值不保留小数
            for i in range(len(P)):
                if p_c_matrix_one[int(P[i]) - 1][int(Q[i]) - 1] == 0:
                    p_c_matrix_one[int(P[i]) - 1][int(Q[i]) - 1] = 1
            p_c_matrix.append(p_c_matrix_one)
    print(p_c_matrix[-2].shape)
    print(p_c_matrix[-2])
    print(len(p_c_matrix))
    return p_c_matrix


# 列压缩格式压缩矩阵,传入要压缩的下标i以及p_c_matrix，则对p_c_matrix[i]进行压缩
def csc_compress(p_c_matrix, i):
    p_c_matrix_csc = sparse.csc_matrix(p_c_matrix[i])
    return p_c_matrix_csc

# p_c_matrix = build_p_c_matrix('data/assist2009_pid/assist2009_pid_train1.csv', 16891, 110)
# print(csc_compress(p_c_matrix,-2))

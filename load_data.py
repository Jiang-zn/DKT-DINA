import numpy as np
import math
from scipy import sparse


class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen

    # data format
    # id, true_student_id
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    # 对每一个学生的数据构建题目-知识点关联矩阵
    # 1.用enumerate遍历每一个学生的所有答题信息
    # 2.对所回答的题目及对应回答的知识点进行标记，构建二维矩阵
    # 3.将每个学生的题目-知识点关联矩阵存储到一个列表中
    # 暂时用不了，因为没有题目序列，只有知识点序列
    # def build_p_c_matrix(self, path):
    #     csv_data = open(path, 'r')
    #     p_c_matrix = []
    #     for lineID, line in enumerate(csv_data):
    #         line = line.strip()
    #         # lineID starts from 0
    #         # 每4行为一个学生的数据，第一行为学生id，第二行为题目id，第三行为知识点序列，第四行为回答序列
    #         if lineID % 4 == 0:
    #             student_id = lineID // 4
    #         if lineID % 4 == 1:
    #             P = line.split(self.separate_char)
    #             if len(P[len(P) - 1]) == 0:
    #                 P = P[:-1]
    #         if lineID % 4 == 1:
    #             Q = line.split(self.separate_char)
    #             if len(Q[len(Q) - 1]) == 0:
    #                 Q = Q[:-1]
    #             # print(len(Q))
    #         elif lineID % 4 == 2:
    #             A = line.split(self.separate_char)
    #             if len(A[len(A) - 1]) == 0:
    #                 A = A[:-1]
    #             # print(len(A),A)
    #
    #             # 题目-知识点关联矩阵(每一个学生都有一个，会不会太冗余？）
    #             # 构建一个二维矩阵，把P中的最大值是这个二维矩阵的行数，Q中的最大值作为这个二维矩阵的列数
    #             max_P = 0
    #             for i in range(len(P)):
    #                 if int(P[i]) > max_P:
    #                     max_P = int(P[i])
    #             max_Q = 0
    #             for i in range(len(Q)):
    #                 if int(Q[i]) > max_Q:
    #                     max_Q = int(Q[i])
    #             # 把矩阵的值都置为0
    #             p_c_matrix_one = np.zeros((max_P, max_Q))
    #             # print(p_c_matrix_one)
    #             # print(p_c_matrix_one.shape)
    #             # 遍历P和Q，把P中的值作为行，Q中的值作为列，把这个二维矩阵中的值置为1，值不保留小数
    #             for i in range(len(P)):
    #                 if p_c_matrix_one[int(P[i]) - 1][int(Q[i]) - 1] == 0:
    #                     p_c_matrix_one[int(P[i]) - 1][int(Q[i]) - 1] = 1
    #             p_c_matrix.append(p_c_matrix_one)
    #     # print(p_c_matrix[-2].shape)
    #     # print(p_c_matrix[-2])
    #     # print(len(p_c_matrix))
    #     return p_c_matrix

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        qa_data = []
        idx_data = []
        p_c_matrix = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 0:
                student_id = lineID // 3
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len(Q[len(Q) - 1]) == 0:
                    Q = Q[:-1]
                # 有多少个知识点
                # print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]
                # 有多少个回答
                # print(len(A),A)

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

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k + 1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            # int(A[i]) is in {0,1}
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    # print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    idx_data.append(student_id)

        # print(p_c_matrix[-2].shape)
        # print(p_c_matrix[-2])
        # print(len(p_c_matrix))
        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        # 知识点q_dataArray，分割序列，统一长度为seqlen
        # 知识点-回答qa_dataArray(回答正确+知识点数n_question)，分割序列，统一长度为seqlen
        # 题目p_dataArray，分割序列，统一长度为seqlen
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_dataArray, qa_dataArray, np.asarray(idx_data), p_c_matrix


class PID_DATA(object):
    def __init__(self, n_question, n_pid, seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        self.n_pid = n_pid
        self.seqlen = seqlen

    # data format
    # id, true_student_id
    # pid1, pid2, ...
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    # 构建总的题目-知识点关联矩阵
    # 不对每一个学生构建自己的题目-知识点关联矩阵
    # 对所回答的题目及对应回答的知识点进行标记，构建二维矩阵
    # 将涉及到的所有题目及对应的知识点关联都存储起来
    def build_p_c_matrix(self, path, ):
        csv_data = open(path, 'r')
        p_c_matrix = []
        for lineID, line in enumerate(csv_data):
            line = line.strip()
            # lineID starts from 0
            # 每4行为一个学生的数据，第一行为学生id，第二行为题目id，第三行为知识点序列，第四行为回答序列
            if lineID % 4 == 0:
                student_id = lineID // 4
            if lineID % 4 == 1:
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]
            if lineID % 4 == 2:
                Q = line.split(self.separate_char)
                if len(Q[len(Q) - 1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # 题目-知识点关联矩阵(每一个学生都有一个，会不会太冗余？）
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
        # print(p_c_matrix[-2].shape)
        # print(p_c_matrix[-2])
        # print(len(p_c_matrix))
        return p_c_matrix

    def load_data(self, path):
        f_data = open(path, 'r')

        q_data = []
        qa_data = []
        p_data = []
        p_c_matrix = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            # 每4行为一个学生的数据，第一行为学生id，第二行为题目id，第三行为知识点序列，第四行为回答序列
            if lineID % 4 == 0:
                student_id = lineID // 4
            if lineID % 4 == 1:
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]
            if lineID % 4 == 2:
                Q = line.split(self.separate_char)
                if len(Q[len(Q) - 1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # 处理数据
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    problem_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k + 1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            problem_sequence.append(int(P[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    p_data.append(problem_sequence)

        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # 将数据转换为ndarrays以提高训练速度
        # 知识点q_dataArray，分割序列，统一长度为seqlen
        # 知识点-回答qa_dataArray(回答正确+知识点数n_question)，分割序列，统一长度为seqlen
        # 题目p_dataArray，分割序列，统一长度为seqlen
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat
        # print(q_dataArray)
        # print(qa_dataArray)
        # print(p_dataArray)
        return q_dataArray, qa_dataArray, p_dataArray

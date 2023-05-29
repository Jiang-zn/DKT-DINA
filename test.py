# import torch
#
# # Step 1: 定义IRT模型示例
# # model = IRTModel(num_students, num_items, num_skills, hidden_size)
#
# # Step 2: 加载模型参数
# model.load_state_dict(torch.load(model_path))
#
# # Step 3: 准备题目-知识点关联矩阵
# item_skill_matrix = torch.tensor(item_skill_matrix)
#
# # Step 4: 把学生答题特征和题目-知识点关联矩阵输入到IRT模型中
# predictions = model(features, item_skill_matrix)

# 读取Qmatrix.csv文件
# import pandas as pd
# Qmatrix = pd.read_csv('Qmatrix.csv', header=None)
# # shape一下
# print(Qmatrix.shape)



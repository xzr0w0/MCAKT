import csv
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

class Data:
    def __init__(self, file, length, concept_num, rows_num, is_train=False, is_test=False, index_split=None):
        csv_file = csv.reader(file, delimiter=',')
        '''
        列表推导式
        筛选模式：[变量(加工后的变量) for 变量 in iterable if 条件]
        双重循环
        int(e)将字符串转为数字
        '''
        # [[题目个数], [题目序列], [答对情况]……]
        rows = [[int(e) for e in i if i != ''] for i in csv_file]

        # 定义题目序列列表和答题序列列表
        q_rows, r_rows = [], []
        # 定义学生数量   注意其使用  更准确意思是学生索引
        student_num = 0

        '''
        列表切片方法
        li[start : end : step]    
        start是切片起点索引，end是切片终点索引，但切片结果不包括终点索引的值。step是步长默认是1。
        '''
        rows_slice_q, rows_slice_r = rows[(rows_num-2)::rows_num], rows[(rows_num-1)::rows_num]
        # print("问题",rows_slice_q.__len__())
        # print("回答",rows_slice_r.__len__())

        '''
        a = [1,2,3]
        b = [4,5,6]
        zipped = zip(a,b)  [(1, 4), (2, 5), (3, 6)] python2会是列表,但python3返回的是个zip的对象
        '''
        zipped = zip(rows_slice_q, rows_slice_r)

        # print('处理前的len(q_rows):' + str(len(rows_slice_q)))
        # print('处理前的max(q_rows):' + str(max([len(e) for e in rows_slice_q])))# len(e) for e in rows_slice_q：表示将rows_slice_q中所有元素的length遍历出来
        '''
        如果是测试数据 则不需要进行index_split划分
        '''
        if is_test:
            for q_row, r_row in zipped:
                num = len(q_row)
                '''
                //为整数除法
                221//100 2
                221/100 2.21
                '''
                n = num // length
                # 这里表示对 zip之后的数据进行切分操作
                # zipped的格式为：zipped = zip(a,b)  [(1, 4), (2, 5), (3, 6)]
                # 因此可以直接划分成q_row和r_row（每200个题添加一次）
                for i in range(n + 1):
                    q_rows.append(q_row[i * length:(i + 1) * length])
                    r_rows.append(r_row[i * length:(i + 1) * length])
        else:
            '''
            如果是训练数据 则需要进行训练集和验证集的划分
            '''
            if is_train:
                for q_row, r_row in zipped:
                    if student_num not in index_split:
                        num = len(q_row)
                        n = num // length
                        for i in range(n + 1):
                            q_rows.append(q_row[i * length:(i + 1) * length])
                            r_rows.append(r_row[i * length:(i + 1) * length])
                    student_num += 1
            else:
                for q_row, r_row in zipped:
                    if student_num in index_split:
                        num = len(q_row)
                        n = num // length
                        for i in range(n + 1):
                            q_rows.append(q_row[i * length:(i + 1) * length])
                            r_rows.append(r_row[i * length:(i + 1) * length])
                    student_num += 1
        q_rows = [row for row in q_rows if len(row) > 3]
        r_rows = [row for row in r_rows if len(row) > 3]

        print('处理后的len(q_rows):' + str(len(q_rows)))
        print('处理后的max(q_rows):' + str(max([len(e) for e in q_rows])))

        self.q_rows = q_rows
        self.r_rows = r_rows
        self.concept_num = concept_num

    '''
    这两个方法是下面做DataLoader需要用到的,不写的话DataLoader无法做批处理

    这两函数是pytorch框架的要求， 用于提取出数据  根据具体需要重写
    '''

    def __getitem__(self, index):

        return list(zip(self.q_rows[index], self.r_rows[index]))

    def __len__(self):
        return len(self.q_rows)


def collate(batch, seq_len):
    '''
    batch是个32长度的列表
    00 [(1,1)(7,1)(1,0)....]
    '''
    # 获得每一条数据的长度放到列表里
    lens = [len(row) for row in batch]
    
    # 获得当前批处理的最长的数据长度
    max_len = max(lens)

    # 填充数据
    padded_data = []
    window_size = 20  # 窗口长度
    stride = 1  # 滑动步长
    for row in batch:
        # 先进行填充
        lenth = len(row)
        padded_data.append([[0, 0, 0]] * (seq_len - lenth + window_size - 1) + [[*e, 1] for e in row] ) 

    padded_and_windowed_data = []

    for padded_row in padded_data:
        lenth = len(padded_row)
        # 利用滑动窗口划分数据
        for t in range(0, lenth - window_size + 1, stride):
            window_data = padded_row[t:t + window_size]
            padded_and_windowed_data.append(window_data)

    # 将数据转换为 Tensor
    batch = torch.tensor(padded_and_windowed_data).cuda()
    Q, Y, S = batch.T  # Q:问题，Y:预测，S:padding
    Q, Y, S = Q.T, Y.T, S.T
    return Y, S, Q



def load_dataset(file_path, batch_size, seq_len, concept_num, rows_num, student_len, val_ratio, shuffle=True, seed = 0):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        graph_type: the type of the concept graph
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    torch.manual_seed(seed)
    test_data = Data(open('%s/test.csv' % file_path, 'r'), seq_len, concept_num,rows_num, is_test=True)
    
    origin_list = [i for i in range(student_len)]
    index_split = random.sample(origin_list, int(val_ratio * len(origin_list)))  # 设置0.1的valid

    train_data = Data(open('%s/train_valid.csv' % file_path, 'r'), seq_len, concept_num, rows_num,
                          is_train=True, index_split=index_split,is_test=False)
    valid_data = Data(open('%s/train_valid.csv' % file_path, 'r'), seq_len, concept_num, rows_num,
                          is_train=False,index_split=index_split,is_test=False)
    # Step 1.2 - Remove users with a single answer 移除只回答了一个问题的user


    # Step 4 - Convert to a sequence per user id and shift features 1 timestep 将每一个用户id的行为做成一个序列
    # 这里的每个list里面的元素都是表示1个学生的一个序列,# question_list[1]就表示学生1的作答序列
   
    window_size = 20
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate(batch,seq_len), drop_last=True)  
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate(batch,seq_len), drop_last=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate(batch,seq_len), drop_last=True)


    return concept_num, train_data_loader, valid_data_loader, test_data_loader

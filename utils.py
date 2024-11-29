
import os
import zipfile
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, auc, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, accuracy_score,r2_score

detach = lambda o: o.cpu().detach().numpy().tolist()  # detach : 分离梯度信息，有些计算时tensor不再需要计算和存储大量的梯度信息，
torch.manual_seed(42)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=True, delta=0, model_file='   .pt', trace_func=print, monitor='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_file = model_file
        self.trace_func = trace_func
        self.monitor = monitor

    # def __call__(self, val_loss, model):
    def __call__(self, val_loss, model, monitor):

        if monitor == 'loss':
            self.monitor = monitor
            score = -val_loss
        if monitor == 'auc':
            self.monitor = monitor
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.monitor == 'loss':
                self.trace_func(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            if self.monitor == 'valid':
                self.trace_func(
                    f'Validation auc increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.model_file)
        self.val_loss_min = val_loss



def load_metr_la_data(node_num):
    print("运行dense graph")
    graph = 1. / (node_num - 1) * np.ones((node_num, node_num))
    np.fill_diagonal(graph, 0)
    graph = torch.from_numpy(graph).float()
    # print("dense graph", graph.shape, graph)
    return graph


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    读取邻接矩阵，并返回度信息
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    A = A.numpy()
    D = np.array(np.sum(A, axis = 1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave




class KTLoss(nn.Module):

    def __init__(self): 
        super(KTLoss, self).__init__()
        self.criterion = nn.BCELoss()
    def forward(self, pred_answers, real_answers, index):
        r"""
        Parameters:
            pred_answers: the correct probability of questions answered at the next timestamp
            real_answers: the real results(0 or 1) of questions answered at the next timestamp
        Shape:
            pred_answers: [batch_size,output_step]
            real_answers: [batch_size,seq_len]
        Return:
        """
        y_true = real_answers[index].float().cpu().detach().numpy()
        y_pred = pred_answers[index].float().cpu().detach().numpy()
        # print('dkvmn shape',DKVMN)
        # print('dkvmn_y shape',DKVMN_Y)
        # print("y_true",y_true)
        # print("y_pred",y_pred)
        try:
             #  计算相关误差
            '''
            fpr 特异性 检测出确实为0的能力 真阳性
            tpr 敏感性 检测出确实为1的能力 真阴性
            thres 阈值
            '''
            fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
            '''
            mse 均方误差 
            mae 平均绝对误差
            越小越好
            '''
            mse_value = mean_squared_error(y_true, y_pred)
            mae_value = mean_absolute_error(y_true, y_pred)
            rmse_value=np.sqrt(mean_squared_error(y_true,y_pred))
            r2_value=r2_score(y_true,y_pred)
            bi_y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
            acc_value = accuracy_score(y_true, bi_y_pred)
            auc_value = auc(fpr, tpr)
        
            # print("auc_value",auc_value)
            # print("acc_value",acc_value)
            '''
            auc(fpr,tpr) Compute Area Under the Curve (AUC) 0.5最差 1最好
            loss 平均化
            '''
        except ValueError as e:
            auc_value, acc_value = -1, -1
            # print(e)



        # calculate NLL loss
        y_pred = y_pred.astype(float)
        y_true = y_true.astype(float)
        # print("y_true",y_true.shape)
        # print("y_pred",y_pred.shape)

        answer_true = real_answers[index].float()
        answer_pred = pred_answers[index].float()
        loss = self.criterion(answer_pred, answer_true) 

        return loss, auc_value, acc_value, mse_value, mae_value, rmse_value,r2_value


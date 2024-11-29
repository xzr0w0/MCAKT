import numpy as np
import time
import random
import argparse
import pickle
import os
import gc
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from processing import *
from model import *
from utils import *
import os
import torch
from torch import detach, nn, eye
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, accuracy_score,r2_score

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

# num_gpus = torch.cuda.device_count()
# print(f"Number of available GPUs: {num_gpus}")

# for i in range(num_gpus):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

parser = argparse.ArgumentParser(description='Script to test DKVMN.')
# 添加参数步骤
parser.add_argument('--no-cuda', action='store_false', default=False, help='Disables CUDA training.')
parser.add_argument('--cuda_device', type=int, default=1, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--data-dir', type=str, default='/home/user10/XZR/dataset/as17', help='Data dir for loading input data./home/user10/XZR/dataset/ /home/Shared_files/assist2009v1')
parser.add_argument('--dataset', type=str, default='as17', help='Name of input data file.  skill_builder_data')
parser.add_argument('--save-dir', type=str, default='logs/as17', help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-dir', type=str, default='', help='Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
parser.add_argument('--num_timesteps_input', type=int, default=20, help='Input timesteps number')
parser.add_argument('--num_timesteps_output', type=int, default=1, help='Output timesteps number')
parser.add_argument('--hid-dim', type=int, default=64, help='Dimension of hidden knowledge states.')
parser.add_argument('--emb-dim', type=int, default=64, help='DimensSion of concept embedding.')
parser.add_argument('--nhead', type=int, default=8, help='')#######
parser.add_argument('--num_decoder_layers', type=int, default=1, help='')#######
parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate (1 - keep probability).')
parser.add_argument('--bias', type=bool, default=True, help='Whether to add bias for neural network layers.')
parser.add_argument('--binary', type=bool, default=True, help='Whether only use 0/1 for results.')
parser.add_argument('--var', type=float, default=1, help='Output variance.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=32, help='Number of samples per batch.')
parser.add_argument('--seq_len', type=int, default=100, help='length of samples per batch.')
parser.add_argument('--train-ratio', type=float, default=0.6, help='The ratio of training samples in a dataset.')
parser.add_argument('--val-ratio', type=float, default=0.25, help='The ratio of validation samples in a dataset.')
parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset or not.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, help='L  R decay factor.')
parser.add_argument('--test', type=bool, default=False, help='Whether to test for existed model.')
parser.add_argument('--test-model-dir', type=str, default='logs/exp09', help='Existed model file dir.')
parser.add_argument('--enable-cuda', action='store_true',help='Enable CUDA')
parser.add_argument('--concept_num_DKVMN', type=int, default=32, help='')
parser.add_argument('--early_stopp_patience', type=int, default=5, help='early_stopp_patience.')
parser.add_argument('--weight_decay', type=int, default=0.01, help='L2 weight_decay.')
parser.add_argument('--q_num', type=int, default=0, help='')
parser.add_argument('--rows_num', type=int, default=4, help='')
parser.add_argument('--student_len', type=int, default=0, help='')
parser.add_argument('--use_att', type=bool, default=True, help='')

args = parser.parse_args()


dataset = args.dataset
if dataset == 'junyi':
    args.q_num = 40
    args.student_len = 72633+24211

if dataset == 'as12':
    args.q_num = 265
    args.student_len =13186+4397

if dataset == 'as09':
    args.q_num = 541
    args.student_len = 4974

if dataset == 'as15':
    args.q_num = 124
    args.student_len = 2368

if dataset == 'as17':
    args.q_num = 3162
    args.student_len = 1708

if dataset == 'statics':
    args.q_num = 1224
    args.student_len = 267
    args.rows_num = 3

if dataset == 'assist2015':
    args.q_num = 102
    args.student_len = 15872
    args.rows_num = 3

if dataset == 'assist2009v1':
    args.q_num = 102
    args.student_len = 3320



args.device = None
if args.no_cuda and torch.cuda.is_available():
    args.device = torch.device('cpu')
else:
    args.device = torch.device('cuda')
    torch.cuda.set_device(args.cuda_device)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print("args.cuda",args.cuda)
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


res_len = 2 

# Save model and meta-data. Always saves in a new sub-folder.
log = None
save_dir = args.save_dir
if args.save_dir:
    exp_counter = 0
    now = datetime.datetime.now()
    # timestamp = now.isoformat()
    timestamp = now.strftime('%Y-%m-%d %H-%M-%S')

    model_file_name = 'ST'

    save_dir = '{}/exp{}/'.format(args.save_dir, model_file_name + timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    meta_file = os.path.join(save_dir, 'metadata.pkl')
    model_file = os.path.join(save_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    log_file = os.path.join(save_dir, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_dir provided!" + "Testing (within this script) will throw an error.")

print(args,file = log)
log.flush()

# load dataset
# dataset_path = os.path.join(args.data_dir, args.data_file)
concept_num, train_data_loader, valid_data_loader, test_data_loader = \
        load_dataset(args.data_dir, args.batch_size, args.seq_len, args.q_num, args.rows_num, args.student_len , args.val_ratio, shuffle=args.shuffle, seed = args.seed)

A = load_metr_la_data(args.concept_num_DKVMN)
A_wave = get_normalized_adj(A)
A_wave = torch.from_numpy(A_wave)
A_wave = A_wave.to(device=args.device)

args.concept_num = concept_num 


model = Model_exp(args.concept_num, args.concept_num_DKVMN, args.hid_dim, args.seq_len, args.nhead,args.num_decoder_layers, args.num_timesteps_input,args.num_timesteps_output, args.dropout,args.use_att).to(device=args.device)

# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

# load model/optimizer/scheduler params
if args.load_dir:
    model_file_name = 'ST'
    model_file = os.path.join(args.load_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    model.load_state_dict(torch.load(model_file))
    optimizer.load_state_dict(torch.load(optimizer_file))
    scheduler.load_state_dict(torch.load(scheduler_file))
    args.save_dir = False

early_stopping = EarlyStopping(patience=args.early_stopp_patience, verbose=True,
                               model_file=model_file)

if args.cuda:
    model = model.cuda()
    kt_loss = KTLoss()


def train(epoch):
    '''
    将模块设置为训练模式
    mode ( bool ) – 是否设置训练模式 ( True) 或评估模式 ( False)。默认值：True。
    '''
    t = time.time()
    loss_train = []
    kt_train = []
    auc_train = []
    acc_train = []
    mse_train = []
    mae_train = []
    rmse_train = []
    r2_train  = []    
    model.train()  # 注意加上此函数
    '''
    二分类交叉熵损失
    预测是否回答正确 0或1 为二分类
    在使用nn.BCELoss需要在该层前面加上Sigmoid函数
    （模型最后就有一个激活函数，这里再sigmoid一遍可能是为了扩大差异）
    '''

    # 取出 batch数据
    for batch_idx, (Y, S, Q) in enumerate(train_data_loader):
        t1 = time.time()
        Y_pred,DKVMN = model(Q, Y, A_wave)

        Y_pred, Y, S = Y_pred[:, -args.num_timesteps_output:], Y[:, -args.num_timesteps_output:], S[:, -args.num_timesteps_output:]
        # 此时P的shape变为(32,199)
        '''
            只提取真实数据的 作loss运算 填充数据不算
            所以通过S转换index矩阵（true,false) ，再P[index]去除。
        '''

        index = S == 1
        index1 = S_DKVMN == 1
        # 梯度下降3步曲
        loss, auc, acc, mse, mae, rmse,r2 = kt_loss(Y_pred, Y, index)  # 1
        if auc != -1 and acc != -1:
            auc_train.append(auc)
            acc_train.append(acc)
            mse_train.append(mse)
            mae_train.append(mae)
            rmse_train.append(rmse)
            r2_train.append(r2) 
        
        print('batch idx: ', batch_idx, 'loss kt: ', loss.item(), 'auc_train: ', auc, 'acc_train: ', acc, 'auc_train1: ',auc1, end=' ')
        if  torch.isnan(torch.tensor(auc)):
            print(Y_pred)
            print(Y)
            break
        
        loss_train.append(float(loss.cpu().detach().numpy()))
        optimizer.zero_grad()  # 记得每次更新梯度
        loss.backward()  # 2
        optimizer.step()  # 3

        del loss
        print('cost time: ', str(time.time() - t1))


    print('Epoch: {:04d}'.format(epoch),
            'loss_train: {:.10f}'.format(np.mean(loss_train)),
            'auc_train: {:.10f}'.format(np.mean(auc_train)),
            'acc_train: {:.10f}'.format(np.mean(acc_train)),
            'auc1_train: {:.10f}'.format(np.mean(auc1_train)),
            'mse_train: {:.10f}'.format(np.mean(mse_train)),
            'mae_train: {:.10f}'.format(np.mean(mae_train)),
            'rmse_train: {:.10f}'.format(np.mean(rmse_train)),
            'r2_train: {:.10f}'.format(np.mean(r2_train)),
            'time: {:.4f}s'.format(time.time() - t))
    print('Epoch: {:04d}'.format(epoch),
            'loss_train: {:.10f}'.format(np.mean(loss_train)),
            'auc_train: {:.10f}'.format(np.mean(auc_train)),
            'acc_train: {:.10f}'.format(np.mean(acc_train)),
            'auc1_train: {:.10f}'.format(np.mean(auc1_train)),
            'mse_train: {:.10f}'.format(np.mean(mse_train)),
            'mae_train: {:.10f}'.format(np.mean(mae_train)),
            'rmse_train: {:.10f}'.format(np.mean(rmse_train)),
            'r2_train: {:.10f}'.format(np.mean(r2_train)),
            'time: {:.4f}s'.format(time.time() - t), file=log)
    log.flush()
    # del loss_train
    # del auc_train
    # del acc_train
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()


# 评估函数  比较与train()的区别
def evaluate(epoch, best_val_loss):

    t = time.time()
    loss_val = []
    kt_val = []
    auc_val = []
    acc_val = []
    mse_val = []
    mae_val = []
    rmse_val = []
    r2_val  = []    
    model.eval()  # 注意加上此函数
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch_idx, (Y, S, Q) in enumerate(valid_data_loader):
            Y_pred = model(Q, Y, A_wave)
            Y_pred, Y, S = Y_pred[:, -args.num_timesteps_output:], Y[:, -args.num_timesteps_output:], S[:, -args.num_timesteps_output:]
            # 此时P的shape变为(32,199)
            '''
                只提取真实数据的 作loss运算 填充数据不算
                所以通过S转换index矩阵（true,false) ，再P[index]去除。
            '''
            index = S == 1
            '''
                detach函数返回一个不带grad的新张量
            因为每个batch里真实样本的个数不一样,所以乘以一个真实样本的个数相当于权重
            '''
        Y_pred, Y, S = Y_pred[:, -args.num_timesteps_output:], Y[:, -args.num_timesteps_output:], S[:, -args.num_timesteps_output:]
        # 此时P的shape变为(32,199)
        '''
            只提取真实数据的 作loss运算 填充数据不算
            所以通过S转换index矩阵（true,false) ，再P[index]去除。
        '''
        # 梯度下降3步曲

        loss_kt, auc, acc, mse, mae, rmse,r2 = kt_loss(Y_pred, Y, index)  # 1
        loss = float(loss_kt.cpu().detach().numpy())
        if auc != -1 and acc != -1:
            auc_val.append(auc)
            acc_val.append(acc)
            loss_val.append(loss)
            mse_val.append(mse)
            mae_val.append(mae)
            rmse_val.append(rmse)
            r2_val.append(r2) 
        del loss
          
            # if auc != -1 and acc != -1:
            #     auc_val.append(auc)
            #     acc_val.append(acc)
            

    print('Epoch: {:04d}'.format(epoch),
            'loss_val: {:.10f}'.format(np.mean(loss_val)),
            'auc_val: {:.10f}'.format(np.mean(auc_val)),
            'acc_val: {:.10f}'.format(np.mean(acc_val)),
            'mse_val: {:.10f}'.format(np.mean(mse_val)),
            'mae_val: {:.10f}'.format(np.mean(mae_val)),
            'rmse_val: {:.10f}'.format(np.mean(rmse_val)),
            'r2_val: {:.10f}'.format(np.mean(r2_val)),
            'time: {:.4f}s'.format(time.time() - t))
    if args.save_dir and np.mean(loss_val) < best_val_loss:
        print('Best model so far, saving...')
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        torch.save(scheduler.state_dict(), scheduler_file)
        print('Epoch: {:04d}'.format(epoch),
                'loss_val: {:.10f}'.format(np.mean(loss_val)),
                'auc_val: {:.10f}'.format(np.mean(auc_val)),
                'acc_val: {:.10f}'.format(np.mean(acc_val)),
                'mse_val: {:.10f}'.format(np.mean(mse_val)),
                'mae_val: {:.10f}'.format(np.mean(mae_val)),
                'rmse_val: {:.10f}'.format(np.mean(rmse_val)),
                'r2_val: {:.10f}'.format(np.mean(r2_val)),
                'time: {:.4f}s'.format(time.time() - t), file=log)
    log.flush()
    res = np.mean(loss_val)
    # del loss_val
    # del auc_val
    # del acc_val
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()
    return res





def test():
    loss_test = []
    auc = []
    acc = []
    mse = []
    mae = []
    rmse = []
    r2 = []
    model.eval()
    model.load_state_dict(torch.load(model_file))
    with torch.no_grad():
        for batch_idx, (Y, S, Q) in enumerate(test_data_loader):
            Y_pred = model(Q, Y, A_wave)
            Y_pred, Y, S = Y_pred[:, -args.num_timesteps_output:], Y[:, -args.num_timesteps_output:], S[:, -args.num_timesteps_output:]
            index = S == 1
            loss_kt, auc, acc, auc1, mse, mae, rmse,r2 = kt_loss(Y_pred, Y, index)  # 1
            loss = float(loss_kt.cpu().detach().numpy())
            loss_test.append(loss)
            del loss
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    
    print('loss_test: {:.10f}'.format(np.mean(loss_test)),
            'auc_test: {:.10f}'.format(np.mean(auc)),
            'acc_test: {:.10f}'.format(np.mean(acc)),
            'mse_test: {:.10f}'.format(np.mean(mse)),
            'mae_test: {:.10f}'.format(np.mean(mae)),
            'rmse_test: {:.10f}'.format(np.mean(rmse)),
            'r2_test: {:.10f}'.format(np.mean(r2)),)
    
    if args.save_dir:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
                'auc_test: {:.10f}'.format(np.mean(auc)),
                'acc_test: {:.10f}'.format(np.mean(acc)),
                'mse_test: {:.10f}'.format(np.mean(mse)),
                'mae_test: {:.10f}'.format(np.mean(mae)),
                'rmse_test: {:.10f}'.format(np.mean(rmse)),
                'r2_test: {:.10f}'.format(np.mean(r2)), file=log)
        log.flush()
    del loss_test
    del auc
    del acc

    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()


if args.test is False:
    # Train model
    print('start training!')
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        train(epoch)
        val_loss = evaluate(epoch, best_val_loss)
        # early_stopping(valid_loss, model)
        early_stopping(val_loss, model, monitor='loss')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_dir:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

test()
if log is not None:
    print(save_dir)
    log.close()
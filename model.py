
'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-07-10 13:22:16
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-04-15 15:47:57
FilePath: /XZR/KSGenerate/model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_
from scipy.stats import pearsonr



torch.manual_seed(7)

class SpatialNetwork(nn.Module):
    def __init__(self,num_timesteps_input, input_dim, output_dim, dropout):
        super(SpatialNetwork, self).__init__()
        self.Theta1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(num_timesteps_input)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adjacency_matrix):
        """
        :param X: Input data of shape [bz, num_timesteps_input, concept_num, dim]
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape [bz, num_timesteps_input, concept_num, dim]
        """
        lfs = torch.einsum("ij,jklm->kilm", [adjacency_matrix, x.permute(2, 1, 0, 3)])  # ([80, 16, 32, 64])
        output = F.relu(torch.matmul(lfs, self.Theta1))
        out1 = output.permute(2, 0, 1, 3)
        out2 = x + self.dropout(out1)
        out3 = self.bn(out2)
        # out4 = torch.tanh(out3)
        return out3
        


class SpatioTemporalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes,num_timesteps_input,num_timesteps_output, dropout):
        super(SpatioTemporalNetwork, self).__init__()
        self.spatial_net = SpatialNetwork(num_timesteps_input,input_dim, output_dim, dropout)
        self.msnet = MSCAN(input_dim, input_dim, num_timesteps_input,num_timesteps_output, hidden_features=512, out_features=output_dim, act_layer=nn.GELU, drop=dropout)
        self.catfully = nn.Linear(hidden_dim*2,hidden_dim)
        self.batch_norm = nn.BatchNorm2d(num_timesteps_input)
        self.time_linear = nn.Linear(num_timesteps_input-2, num_timesteps_input)
        self.fully = nn.Sequential(
            nn.Tanh(),
            nn.Linear(num_timesteps_input * hidden_dim, num_timesteps_output),
            nn.Sigmoid()
            )
        
        self.net1 = nn.Linear(num_timesteps_input, num_timesteps_output)
        self.net2 = nn.Linear(num_nodes * hidden_dim, num_nodes)
        


    def forward(self, input, A_hat): #[bz, num_timesteps_input, concept_num, dim]
        spatial_output = self.spatial_net(input, A_hat)  # [2592, 19, 32, 64]
        temporal_output = self.msnet(input)  # [2592, 19, 32, 64]
        x3 = spatial_output + temporal_output
        x4 =  self.batch_norm(x3).permute(0,2,1,3)
        out = self.fully(x4.reshape((x4.shape[0], x4.shape[1], -1))).permute(0, 2, 1)
        # print("out",out)
        return out

class DWConv(nn.Module):
    def __init__(self, dim=512):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, (3,1), (1,1), (1,0), bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
    
    

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(FFN, self).__init__()
        
        # Temporal convolutional layer
        self.temporal_conv = nn.Conv2d(in_features, out_features * 2, 1, stride=1, padding=0)
        
    def forward(self, x):
        # Temporal convolution
        temporal_conv_out = self.temporal_conv(x)
        
        # Split the output into two parts: gated and processed
        gated, processed = temporal_conv_out.chunk(2, 1)
        
        # Apply gating mechanism (sigmoid activation)
        gated = torch.sigmoid(gated)
        
        # Element-wise multiplication of processed and gated parts
        output = processed * gated
        
        return output
    

class MSCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (19, 1), padding=(9, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

        self.scale_weights = nn.Parameter(torch.ones(4))

        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.scale_weights[0])
        self.scale_weights.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # u = x.clone()
        u = self.conv0(x)
        attn_1 = self.conv0_2(u)
        attn_2 = self.conv1_2(u)
        attn_3 = self.conv2_2(u)
        scale_weights = F.softmax(self.scale_weights, dim=0)  # 使用softmax将权重归一化
        attn = u * scale_weights[0] + attn_1 * scale_weights[1] + attn_2 * scale_weights[2] + attn_3 * scale_weights[3]

        attn = self.conv3(attn)

        return attn * u
    
class AttentionModule(nn.Module):   
    def __init__(self, dim):
        super().__init__()
        self.att = MSCA(dim)
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.att(x)
        x = self.fc2(x)
        return x

class MSCAN(nn.Module):
    def __init__(self, dim, in_features, in_step, out_step,  hidden_features=None, out_features=None,act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.att = AttentionModule(dim)
        self.fnn = FFN(in_features, hidden_features, out_features, act_layer, drop)
        self.bn1 = nn.BatchNorm2d(in_features)
        self.bn2 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.act = act_layer()
    def forward(self, x):
        x = x.permute(0,3,1,2)
        u = x.clone()
        x1 = self.att(x) + u
        v = x1.clone()
        x2 = self.bn2(x1)
        x2 = self.fnn(x2)
        out = x2 + v
        out = x2.permute(0,2,3,1)
        return out   # [2592, 19, 32, 64]



class KSGenerate(nn.Module):

    def __init__(self, q_num, hidden_num, concept_num, num_timesteps_input, num_timesteps_output):
        super(KSGenerate, self).__init__()
        self.q_num = q_num
        self.hidden_num, self.concept_num = hidden_num, concept_num
        self.num_timesteps_input, self.num_timesteps_output = num_timesteps_input, num_timesteps_output
        self.q_embedding = nn.Embedding(q_num, hidden_num)

        self.key_linear = nn.Linear(hidden_num, concept_num, bias=False)

        self.x_embedding = nn.Embedding(2 * q_num, hidden_num)
        self.e_fnn = nn.Sequential(
            nn.Linear(hidden_num, hidden_num),
            nn.Sigmoid(),
        )
        self.a_fnn = nn.Sequential(
            nn.Linear(hidden_num, hidden_num),
            nn.Tanh(),
        )
        self.mix_fnn = nn.Sequential(
            nn.Linear(num_timesteps_input+num_timesteps_output, num_timesteps_input),
            nn.Tanh(),
        )
        
        self.criterion = nn.BCELoss()

    def forward(self, Q, Y):
        # Q [bz, length], Y [bz, length], S [bz,length]
        q_num = self.q_num
        X = Q + q_num * (1 - Y) # [bz, length, 2q_num]

        batch_size, length = Q.shape

        hidden_num, concept_num = self.hidden_num, self.concept_num

        Q_emb = self.q_embedding(Q)     # [bz, length, hid]
        # 意思是在第二维度上softmax
        Lambda = F.softmax(self.key_linear(Q_emb), 2)  # [bz, length, concept_num] concept维度被softmax了

        N = torch.mean(Lambda, dim=0).cpu().detach().numpy()
        adjacency_matrix = torch.zeros((N.shape[1], N.shape[1]))
        for i in range(N.shape[1]):
            for j in range(N.shape[1]):
                adjacency_matrix[i, j] = pearsonr(N[:, i], N[:, j])[0]
        adjacency_matrix = adjacency_matrix.cuda()

        X = self.x_embedding(X)   # [bz, length, hid]

        Phi = torch.zeros((batch_size, concept_num, hidden_num), device=Q.device) # [batch_size, concept_num, hidden_num]

        P = []
        for t in range(self.num_timesteps_input):
            Qt = Q_emb[:, t, :].unsqueeze(1)  #[bz, 1, hid]
            Lt = Lambda[:, t, :].unsqueeze(1)    #[bz, 1, concept]
            Xt = X[:, t, :].unsqueeze(1)    #[bz, 1, hid]

            P.append(Phi) # [length, bz, concept, hid]
   
            Ft = self.f_fnn(Xt) # sigmoid操作
            last_Phi = Phi * (1 - torch.bmm(Lt.transpose(1, 2), Ft))  # [bz, concept, hid]
            At = self.a_fnn(Xt)  # #[bz, 1, hid]
            Phi = last_Phi + torch.bmm(Lt.transpose(1, 2), At)  # [bz, concept, hid]

        P_tensor = torch.tensor([item.cpu().detach().numpy() for item in P]).cuda()  #[length_input, bz, concept, hid]
        Qx = L[:,:self.num_timesteps_input,:]   
        Qy = L[:,-self.num_timesteps_output:,:] 
        mix_ht =  P_tensor.permute(1,0,2,3) 
        mix_ht =  torch.cat((mix_ht, Qy.unsqueeze(3).expand(-1, -1, -1, hidden_num)),dim=1)
        mix_ht = self.mix_fnn(mix_ht.permute(0,2,3,1)).permute(0,3,1,2)

        return mix_ht,adjacency_matrix, Qy
    

class DecoderEmbedding(nn.Module):
    def __init__(self, q_num, length,d_model):
        super(DecoderEmbedding, self).__init__()
        self.seq_len = length
        self.exercise_embed = nn.Embedding(q_num, d_model)
        self.position_embed = nn.Embedding(length, d_model)

    def forward(self, exercises): # [bz, num_timesteps_output, concept_num, dim]
        e = self.exercise_embed(exercises.long())
        seq = torch.arange(self.seq_len).cuda().unsqueeze(0)
        p = self.position_embed(seq)
        return e+p



class MyMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim 
        self.head_dim = embed_dim // num_heads 
        self.kdim = self.head_dim
        self.vdim = self.head_dim

        self.num_heads = num_heads
        self.dropout = dropout

        assert self.head_dim * num_heads == self.embed_dim,


        self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))  # embed_dim = kdim * num_heads
        self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))  
        self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim)) 

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


        self._reset_parameters()

    def _reset_parameters(self):
        """
        以特定方式来初始化参数
        :return:
        """
        stdv = 1. / math.sqrt(self.q_proj_weight.shape[1])
        self.q_proj_weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.k_proj_weight.shape[1])
        self.k_proj_weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.v_proj_weight.shape[1])
        self.v_proj_weight.data.uniform_(-stdv, stdv)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        num_heads= self.num_heads
        dropout_p=self.dropout
        out_proj_weight=self.out_proj.weight
        out_proj_bias=self.out_proj.bias
        training=self.training
        q_proj_weight=self.q_proj_weight
        k_proj_weight=self.k_proj_weight
        v_proj_weight=self.v_proj_weight

        q = F.linear(query, q_proj_weight)
        #  [tgt_len,batch_size, embed_dim] x [embed_dim,kdim * num_heads] = [tgt_len,batch_size,kdim * num_heads]

        k = F.linear(key, k_proj_weight)
        # [src_len, batch_size, embed_dim] x [embed_dim, kdim * num_heads] = [src_len, batch_size, kdim * num_heads]

        v = F.linear(value, v_proj_weight)
        # [src_len, batch_size, embed_dim] x [embed_dim, vdim * num_heads] = [src_len, batch_size, vdim * num_heads]
        tgt_len, bsz, embed_dim = query.size()  # [tgt_len,batch_size, embed_dim]
        src_len = key.size(0)
        head_dim = embed_dim // num_heads  # num_heads * head_dim = embed_dim
        scaling = float(head_dim) ** -0.5
        q = q * scaling  # [query_len,batch_size,kdim * num_heads]

        if attn_mask is not None:  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len,src_len]
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')

            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            # 现在 atten_mask 的维度就变成了3D

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        # [batch_size * num_heads,tgt_len,kdim]
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,kdim]
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,vdim]
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))


        if attn_mask is not None:
            # attn_output_weights += attn_mask  
            attn_output_weights=attn_output_weights.masked_fill(attn_mask,float('-inf')) # [batch_size * num_heads, tgt_len, src_len]
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            # 变成 [batch_size, num_heads, tgt_len, src_len]的形状
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # 扩展维度，从[batch_size,src_len]变成[batch_size,1,1,src_len]
                float('-inf'))  #
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,
                                                        src_len)  # [batch_size * num_heads, tgt_len, src_len]

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # [batch_size * num_heads, tgt_len, src_len]
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

        Z = F.linear(attn_output, out_proj_weight, out_proj_bias)
        # 这里就是多个z  线性组合成Z  [tgt_len,batch_size,embed_dim]
        return Z, attn_output_weights.sum(dim=1) / num_heads  # average attention weights over heads


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()

        self.self_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.multihead_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)


        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, tgt, memory,tgt_att_mask=None, memory_att_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
    
        tgt2 = self.self_attn(tgt, tgt, tgt,  # [tgt_len,batch_size, embed_dim]
                              attn_mask=tgt_att_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)  # 接着是残差连接
        tgt = self.norm1(tgt)  # [tgt_len,batch_size, embed_dim]

        tgt2 = self.multihead_attn(tgt,memory,memory,  # [tgt_len, batch_size, embed_dim]
                                   attn_mask=memory_att_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
       
        tgt = tgt + self.dropout2(tgt2)  # 残差连接
        tgt = self.norm2(tgt)  # [tgt_len, batch_size, embed_dim]

        tgt2 = self.activation(self.linear1(tgt))  # [tgt_len, batch_size, dim_feedforward]
        tgt2 = self.linear2(self.dropout(tgt2))  # [tgt_len, batch_size, embed_dim]

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt



class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, dropout = 0):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory,tgt_att_mask=None, memory_att_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        output = tgt  # [tgt_len,batch_size, embed_dim]

        for mod in self.layers:  # 这里的layers就是N层解码层堆叠起来的
            output1 = mod(output, memory,
                         tgt_att_mask=tgt_att_mask,
                         memory_att_mask=memory_att_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask
                         )

        if self.norm is not None:
            output2 = self.norm(output1)

        return output2 

class Model_exp(nn.Module):
    def __init__(self, q_num,concept_num, d_model, length,nhead,num_decoder_layers, num_timesteps_input,num_timesteps_output, dropout,use_att):
        super(Model_exp, self).__init__()
        self.use_att = use_att
        self.num_timesteps_output = num_timesteps_output
        self.KSGenerate = KSGenerate(q_num, d_model, concept_num, num_timesteps_input, num_timesteps_output)
        self.stnet = SpatioTemporalNetwork(d_model, d_model, d_model, concept_num, num_timesteps_input,num_timesteps_output,dropout)
        self.enc_fc = nn.Sequential(
            nn.Linear(concept_num,128), nn.ReLU(),
            nn.Linear(128,d_model),nn.Sigmoid()
        )
        self.enc_emb = nn.Embedding(q_num, d_model)
  #  ================ decoder =====================
        self.decoder_embedding = DecoderEmbedding(q_num=q_num,
                                                  length=num_timesteps_output, d_model=d_model) #试题种类，
        decoder_layer = MyTransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = MyTransformerDecoder(decoder_layer=decoder_layer,num_layers= num_decoder_layers, norm=decoder_norm, dropout=dropout)

  #  ================ predictor =====================
        self.out_fc = nn.Sequential(
            nn.Linear(d_model,128), nn.Tanh(),
            nn.Linear(128,1),nn.Sigmoid()
        )

        self._reset_parameters()

        self.FCs = nn.Sequential(
            nn.Linear(2 * concept_num, concept_num),
            nn.Tanh(),
            nn.Linear(concept_num, 1),
            nn.Sigmoid(),
        )

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        """
        初始化
        """
        for p in self.parameters():
            # print(p)
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self,Q, Y, A_hat):
        # print(Q)
        Q_in = Q[:, -self.num_timesteps_output:] 
        mix_ht,weighted_adjacency_matrix,Qy,KSGenerate = self.KSGenerate(Q,Y)
        encoder_output = self.stnet(mix_ht, weighted_adjacency_matrix) # [32, 20, 16]
        if self.use_att:
            encoder_output = self.enc_fc(encoder_output)
            #  ================ decoder =====================
            dec = self.decoder_embedding(Q_in)
            output = self.decoder(tgt=dec.permute(1, 0, 2), memory=encoder_output.permute(1, 0, 2))
            output = output.permute(1, 0, 2)
            #  ================ predictor =====================
            out=output-dec
            out = self.out_fc(out)
            P=out.squeeze(-1)
        else:
            P = (encoder_output * Qy).sum(2)
        return P, KSGenerate
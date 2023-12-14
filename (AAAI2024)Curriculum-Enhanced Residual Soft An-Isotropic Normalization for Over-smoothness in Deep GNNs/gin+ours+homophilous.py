import os.path as osp
import os
from tkinter import Y
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
from dataload import load_data, load_ogbn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import math
import dgl
import dgl.nn as dglnn
import torch
import torch as th
from torch import nn
import dgl.function as fn
from dgl.base import DGLError
from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm



def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def f(hidden_layer,hidden_channel,param_prop,param_before,param_first,param_ani,lr,weight_decay,lr_decay,epoch,dropout,PPR_num,lamda1,lamda2,gamma,knn_num,label_beta,d_0_rate,an_beta):

    def anisotropy(B):
        N,d=B.size()
        d_0=int(d*d_0_rate)
        S, U = torch.linalg.eigh(B.T @ B)
        S = S.float()
        U = U.float()
        t = B @ U
        for i in range(d):
            if i<=d_0:
                S[-1-i]=S[-1-i]**((-1./2)*an_beta)
            else:
                S[-1-i]*=0
        ans = t @ torch.diag(S) @ U.T
        return ans


    def get_H(p):
        eps = 0.0000001
        return -torch.sum(p*torch.log(p+eps),dim=1)

    def get_consistence(Z):
        consistence = get_H(Z)
        consistence = 1 - consistence / math.log(Z.size()[0])
        return consistence

    def cross_entropy(outputs, targets):
        consistence = get_consistence(targets)
        x = outputs * consistence.reshape(-1,1)   
        loss = torch.mean(torch.sum(-x * targets, dim=1))
        return loss**gamma2

    def crossentropy_loss(x, y):
        x = -x
        assert not (y<0).any()
        return torch.mean(torch.sum(x*y, dim=1))

  
    class SGConv(nn.Module):
        def __init__(self,
                    in_feats,
                    out_feats,
                    k=1,
                    cached=False,
                    bias=True,
                    norm=None,
                    allow_zero_in_degree=False):
            super(SGConv, self).__init__()
            self.fc = nn.Linear(in_feats, out_feats, bias=bias)
            self._cached = cached
            self._cached_h = None
            self._k = k
            self.norm = norm
            self._allow_zero_in_degree = allow_zero_in_degree
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)

        def set_allow_zero_in_degree(self, set_value):
            self._allow_zero_in_degree = set_value

        def forward(self, graph, feat, edge_weight=None):
            with graph.local_scope():
                if not self._allow_zero_in_degree:
                    if (graph.in_degrees() == 0).any():
                        raise DGLError('There are 0-in-degree nodes in the graph, '
                                    'output for those nodes will be invalid. '
                                    'This is harmful for some applications, '
                                    'causing silent performance regression. '
                                    'Adding self-loop on the input graph by '
                                    'calling `g = dgl.add_self_loop(g)` will resolve '
                                    'the issue. Setting ``allow_zero_in_degree`` '
                                    'to be `True` when constructing this module will '
                                    'suppress the check and let the code run.')

                msg_func = fn.copy_u("h", "m")
                feat_total=feat.unsqueeze(0)
                feat_first=feat
                if edge_weight is not None:
                    graph.edata["_edge_weight"] = EdgeWeightNorm(
                        'right')(graph, edge_weight)
                    msg_func = fn.u_mul_e("h", "_edge_weight", "m")

                if self._cached_h is not None:
                    feat = self._cached_h
                else:
                    if edge_weight is None:
                  
                        degs = graph.in_degrees().float().clamp(min=1)
                        norm = th.pow(degs, -0.5)
                        norm = norm.to(feat.device).unsqueeze(1)
        
                    for _ in range(self._k):
                        if edge_weight is None:
                            feat = feat * norm
                        graph.ndata['h'] = feat
                        graph.update_all(msg_func,
                                        fn.sum('m', 'h'))
                        feat = graph.ndata.pop('h')

                        feat_total=torch.cat((feat_total,feat.unsqueeze(0)),dim=0)

                        if edge_weight is None:
                            feat = feat * norm

                    if self.norm is not None:
                        feat = self.norm(feat)

                    if self._cached:
                        self._cached_h = feat
                
                return feat_total
    
    
    class Pre_Train(torch.nn.Module):
        def __init__(self):
            super(Pre_Train,self).__init__()   
            self.conv=SGConv(int(max(data.y)) + 1,1,PPR_num-1)   
       
        def forward(self,data,emb,dataset,knn_num):
            N,d=data.x.size()
            edge_index,edge_weight=torch.load("./Knngraph_embeeding/"+"knngraph_"+dataset+"_"+str(knn_num)+"neighbors")
            edge_index.cuda(),edge_weight.cuda()
            
            edge_index=dgl.graph((edge_index[0],edge_index[1]))
            edge_index = dgl.add_self_loop(edge_index)
                        
            edge_weight=F.relu(edge_weight)**gamma

            ones=torch.ones(N).cuda()
            edge_weight=torch.cat((edge_weight,ones),dim=0)
            x=F.one_hot(data.y).float().cuda()
            x[~data.train_mask,:]=F.softmax(emb[~data.train_mask,:],dim=1)
            label=self.conv(edge_index,x,edge_weight=edge_weight)
            return label


    class Net(torch.nn.Module):
        def __init__(self):
            super(Net,self).__init__()   
            self.gcn_layers=torch.nn.ModuleList()
            self.gcn_layers.append(dglnn.GINConv(th.nn.Linear(data.num_node_features,hidden_channel),aggregator_type='mean'))         
            for i in range(hidden_layer-2):
                self.gcn_layers.append(dglnn.GINConv(th.nn.Linear(hidden_channel,hidden_channel),aggregator_type='mean'))
            self.conv2 = dglnn.GINConv(th.nn.Linear(hidden_channel, int(max(data.y) + 1)),aggregator_type='mean')      
       
        def forward(self,data):
            x,edge_index=data.x, data.edge_index 
            edge_index=dgl.graph((edge_index[0],edge_index[1]), num_nodes=data.x.size(0))
            x_origin=0
            x_first=0
            for i,layer in enumerate(self.gcn_layers) :       
                if i>0:  
                    x_origin=x_origin*lamda1+x
                    x_first+=(lamda2**(i-1))*x
                    s=param_prop+param_before+param_first
                    x=(param_prop/s)*layer(edge_index,x)+(param_before/s)*x_origin+(param_first/s)*x_first
                    
                else:
                    x=layer(edge_index,x)
                x=x-torch.mean(x, dim=0, keepdims=True)
                x=(1-param_ani)*x+param_ani*anisotropy(x.detach())
                x=F.relu(x)
                x=F.dropout(x,p=dropout,training=self.training)
            x=self.conv2(edge_index,x)
            x=F.dropout(x,p=dropout,training=self.training)
            return F.log_softmax(x,dim=1)
   
   
    seed=1
    set_seed(seed)
    dataset = 'Cora'
    data=load_data(dataset, which_run=0)   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    data = data.to(device) 
    model = Net().to(device)
    
    emb=(torch.exp(torch.load("./Embeeding/"+dataset))).cuda()
    model2 = Pre_Train().to(device)
    labels =model2(data,emb.detach(),dataset,knn_num)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[lr_decay],gamma = 0.5)
   
   
    def train(i): 
        model.train() 
        optimizer.zero_grad()
        gap=epoch//PPR_num
        begin=math.ceil((epoch)/gap)-1
        pos=math.ceil((epoch-i)/gap)-1
        output=model(data)
        if i<epoch:
            if pos<begin:
                label=label_beta*labels[pos]+(1-label_beta)*labels[pos+1]
                loss_train=crossentropy_loss(output,label)
            else:  
                loss_train=crossentropy_loss(output,labels[pos])
        else:
            loss_train=crossentropy_loss(output,labels[0])
        loss_train.backward()
        optimizer.step()
        return loss_train.item()

    @torch.no_grad()
    def test():
        model.eval()
        logits = model(data)
        loss_val = F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()
        for _, mask in data('train_mask'):    
            pred = logits[mask].max(1)[1]
            train_accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        for _, mask in data('val_mask'):    
            pred = logits[mask].max(1)[1]
            val_accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        for _, mask in data('test_mask'):    
            pred = logits[mask].max(1)[1]
            test_accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        return loss_val,val_accs,test_accs,train_accs

    best_val_loss = 9999999
    test_acc = 0
    bad_counter = 0
    best_epoch = 0
    max_test=0
    max_val=0
    max_val_test=0
    max_max_val_test = -1
    max_max_val_test_epoach = -1
    for i in tqdm(range(epoch)):     
        loss_tra = train(i)
        scheduler.step()
        loss_val,acc_val_tmp,acc_test_tmp,acc_train_tmp= test()
        if acc_test_tmp>max_test:
            max_test=max(max_test,acc_test_tmp)
        if acc_val_tmp==max_val:
            max_val_test=max(max_val_test,acc_test_tmp)  
        if acc_val_tmp>max_val:
            max_val=acc_val_tmp
            max_val_test=acc_test_tmp
        if max_val_test > max_max_val_test:
            max_max_val_test_epoach = i + 1
            max_max_val_test = max_val_test
        log = 'Epoch: {:03d}, Train loss: {:.4f},Train acc:{:.4f}, Val loss: {:.4f}, Val acc:{:.4f},Test acc: {:.4f}'  
        print(log.format(i, loss_tra,acc_train_tmp, loss_val, acc_val_tmp,acc_test_tmp))
        print("max_test:{:.4f}".format(max_test),"max_val_test:{:.4f}".format(max_val_test))
        print("max_val_test_history:{:.4f}".format(max_max_val_test))
        
       
    return max_val_test
 



d={'hidden_layer': 32, 'hidden_channel': 64, 'param_prop': 0.9, 'param_before': 0.2, 'param_first': 0.5, 'param_ani': 0.01, 'PPR_num': 10, 'lamda1': 0.4, 'lamda2': 0.3, 'gamma': 0.1,'label_beta': 0.8, 'epoch': 300, 'knn_num': 7, 'weight_decay': 0.0001, 'lr_decay': 100, 'lr': 0.01, 'dropout': 0.7, 'an_beta': 1.0, 'd_0_rate': 0.99}
f(**d)

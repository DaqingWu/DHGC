# -*- encoding: utf-8 -*-

import os
import random
from loguru import logger
import numpy as np
import pickle
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import paras
from model import DHGC
from utils import sinkhorn, pdf_norm, estimate_kappa, evaluation

torch.cuda.set_device(paras.args.gpu)

logger.add(paras.args.log_save_path, rotation="500 MB", level="INFO")
logger.info(" --gamma " + str(paras.args.gamma) + " --lambdas " + str(paras.args.lambdas))

seed = 2021

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

x = np.loadtxt('./data/{}.txt'.format(paras.args.name), dtype=float)
if paras.args.name in ['amazon']:
    x = torch.tensor(x, dtype=torch.float)
    x = torch.nn.functional.normalize(x, p=1, dim=1)
else:
    x = torch.tensor(x, dtype=torch.float)
y = np.loadtxt('./data/{}_label.txt'.format(paras.args.name), dtype=int)

x_ = torch.nn.functional.normalize(x, p=2, dim=1)
adj_f = torch.mm(x_, x_.T)

edge_index = torch.tensor(np.loadtxt(paras.args.graph_path, dtype=int), dtype=torch.long).T
adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), [x.shape[0], x.shape[0]]).to_dense()
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj_self_loop = adj+torch.eye(x.shape[0])

model = DHGC(dim_input=paras.args.d_input, dims_encoder=paras.args.dims_en, dims_decoder=paras.args.dims_de, pretrain_model_load_path=paras.args.model_path).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=paras.args.lr, weight_decay=paras.args.wd)

with torch.no_grad():
    z, x_hat = model.forward(x.cuda(), adj_self_loop.cuda())
    kmeans = KMeans(n_clusters=paras.args.n_cluster, n_init=20, random_state=seed).fit(z.cpu().numpy())
    centers = torch.nn.functional.normalize(torch.tensor(kmeans.cluster_centers_), p=2, dim=1).cuda()
    pseudo_labels = torch.LongTensor(kmeans.labels_).cuda()

acc_list = []
nmi_list = []
ari_list = []
f1_list = []
v_list = []
v_max = 0.0
best_epoch = 0

for epoch in range(1, paras.args.epochs+1):
    z, x_hat = model.forward(x.cuda(), adj_self_loop.cuda())
    centers = centers.detach()

    with torch.no_grad():
        if epoch == 1:
            dist = 2 - 2*torch.sum(torch.mul(z, centers[pseudo_labels]), dim=1)
            variance = torch.tensor([torch.mean(dist[torch.nonzero(pseudo_labels==i).squeeze()]) for i in range(paras.args.n_cluster)])
            kappas = 1/variance
            mixture_cofficences = torch.tensor([torch.sum(pseudo_labels==i)/x.shape[0] for i in range(paras.args.n_cluster)]).cuda()
            normalize_constants = pdf_norm(paras.args.dims_en[-1], kappas)
        else:
            kappas = estimate_kappa(paras.args.dims_en[-1], kappas)
            mixture_cofficences = torch.tensor([torch.sum(q_max_index==i)/x.shape[0] for i in range(paras.args.n_cluster)]).cuda()
            normalize_constants = pdf_norm(paras.args.dims_en[-1], kappas)

    p = model.prediction(kappas.cuda(), centers, normalize_constants.cuda(), mixture_cofficences.cuda())

    with torch.no_grad():
        if epoch == 1:
            q = torch.tensor(sinkhorn(p.cpu().numpy(), paras.args.lambdas, torch.ones(x.shape[0]).numpy(), torch.tensor([torch.sum(pseudo_labels==i) for i in range(paras.args.n_cluster)]).numpy())).float().cuda()
        else:
            q = torch.tensor(sinkhorn(p.cpu().numpy(), paras.args.lambdas, torch.ones(x.shape[0]).numpy(), torch.tensor([torch.sum(q_max_index==i) for i in range(paras.args.n_cluster)]).numpy())).float().cuda()
        q_max, q_max_index = torch.max(q, dim=1)

    adj_pred = torch.mm(z, z.T)
    loss_a = torch.nn.functional.mse_loss(adj_pred.view(-1), adj_f.cuda().view(-1))
    loss_x = torch.nn.functional.mse_loss(x_hat, x.cuda())

    loss_p = -torch.mean(torch.sum(torch.mul(kappas[q_max_index].unsqueeze(1).cuda(), torch.mul(z, centers[q_max_index])), dim=1))

    loss = loss_x + loss_a + paras.args.gamma * loss_p

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        q_sorted, q_sorted_index = torch.topk(input=q, k=x.shape[0], dim=0, largest=True, sorted=True)
        
        tao_pull = q_sorted[(mixture_cofficences*x.shape[0]).long()].diag()
        tao_push = q_sorted[(mixture_cofficences*2*x.shape[0]).long()].diag()

        Q = q.cpu().numpy().argmax(1)
        acc_q, nmi_q, ari_q, f1_macro_q = evaluation(y, Q)
        v_q = acc_q + nmi_q + ari_q + f1_macro_q
        logger.info('Epoch {}/{} Train ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}, V:{:.4f}'.format(epoch, paras.args.epochs, acc_q, nmi_q, ari_q, f1_macro_q, v_q))

        acc_list.append(acc_q)
        nmi_list.append(nmi_q)
        ari_list.append(ari_q)
        f1_list.append(f1_macro_q)
        v_list.append(v_q)
        if v_q >= v_max:
            v_max = max(v_list)
            best_epoch = epoch
    
        pull_step = torch.where(q >= tao_pull, torch.ones_like(q)*paras.args.eta, torch.zeros_like(q))
        push_step = torch.where((q < tao_pull) + (q >= tao_push), torch.ones_like(q)*paras.args.eta, torch.zeros_like(q))
        delta = z.repeat(paras.args.n_cluster, 1, 1) - centers.unsqueeze(1)
        centers = torch.nn.functional.normalize(centers + \
                                                torch.sum(torch.mul(delta, torch.mul(pull_step*(50/(50+epoch)), q).T.unsqueeze(2)), dim=1) - \
                                                torch.sum(torch.mul(delta, torch.mul(push_step*(50/(50+3*epoch)), q).T.unsqueeze(2)), dim=1), p=2, dim=1)
                   
logger.info('Best Epoch {}/{}'.format(best_epoch, paras.args.epochs))
logger.info('ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(acc_list[best_epoch-1], nmi_list[best_epoch-1], ari_list[best_epoch-1], f1_list[best_epoch-1]))
            

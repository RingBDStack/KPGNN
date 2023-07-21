"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import time
import sys
sys.path.insert(0, "../../")
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl_dataset import TwitterDataset
from SimpleHGN import RGAT

import sys
sys.path.append('../../')
import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import myGAT
import dgl

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_model_DBLP(args):

    if args.dataset == "twitter":
        dataset = TwitterDataset("./incremental_graph/", split=0)
    else:
        raise ValueError()

    g = dataset[0]
    edge2type = {}
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    for _ in range(args.repeat):
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()

            logits = net(features_list, e_feat)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss.item(), t_end-t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                logits = net(features_list, e_feat)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits = net(features_list, e_feat)
            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{args.run}.txt")
            pred = onehot[pred]
            print(dl.evaluate(pred))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--run', type=int, default=1)

    args = ap.parse_args()
    os.makedirs('checkpoint', exist_ok=True)
    run_model_DBLP(args)

# def main(args):
#     if args.dataset == "twitter":
#         dataset = TwitterDataset("./incremental_graph/", split=0)
#     else:
#         raise ValueError()

#     g = dataset[0]
#     category = dataset.predict_category
#     num_classes = dataset.num_classes
#     train_mask = g.nodes[category].data.pop("train_mask")
#     test_mask = g.nodes[category].data.pop("test_mask")
#     train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
#     test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
#     labels = g.nodes[category].data.pop("labels")

#     category_id = len(g.ntypes)
#     for i, ntype in enumerate(g.ntypes):
#         if ntype == category:
#             category_id = i

#     # split dataset into train, validate, test
#     if args.validation:
#         val_idx = train_idx[: len(train_idx) // 5]
#         train_idx = train_idx[len(train_idx) // 5 :]
#     else:
#         val_idx = train_idx

#     # check cuda
#     use_cuda = args.gpu >= 0 and th.cuda.is_available()
#     if use_cuda:
#         th.cuda.set_device(args.gpu)
#         g = g.to("cuda:%d" % args.gpu)
#         labels = labels.cuda()
#         train_idx = train_idx.cuda()
#         test_idx = test_idx.cuda()

#     # create model
#     model = EntityClassify(
#         g,
#         args.n_hidden,
#         num_classes,
#         num_bases=args.n_bases,
#         num_hidden_layers=args.n_layers - 2,
#         dropout=args.dropout,
#         use_self_loop=args.use_self_loop,
#     )

#     if use_cuda:
#         model.cuda()

#     # optimizer
#     optimizer = th.optim.Adam(
#         model.parameters(), lr=args.lr, weight_decay=args.l2norm
#     )

#     # training loop
#     print("start training...")
#     dur = []
#     model.train()
#     for epoch in range(args.n_epochs):
#         optimizer.zero_grad()
#         if epoch > 5:
#             t0 = time.time()
#         logits = model()[category]
#         print(logits[train_idx].shape, labels[train_idx].shape)
#         loss = F.cross_entropy(logits[train_idx], labels[train_idx])
#         loss.backward()
#         optimizer.step()
#         t1 = time.time()

#         if epoch > 5:
#             dur.append(t1 - t0)
#         train_acc = th.sum(
#             logits[train_idx].argmax(dim=1) == labels[train_idx]
#         ).item() / len(train_idx)
#         val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
#         val_acc = th.sum(
#             logits[val_idx].argmax(dim=1) == labels[val_idx]
#         ).item() / len(val_idx)
#         print(
#             "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".format(
#                 epoch,
#                 train_acc,
#                 loss.item(),
#                 val_acc,
#                 val_loss.item(),
#                 np.average(dur),
#             )
#         )
#     print()
#     if args.model_path is not None:
#         th.save(model.state_dict(), args.model_path)

#     model.eval()
#     logits = model.forward()[category]
#     test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
#     test_acc = th.sum(
#         logits[test_idx].argmax(dim=1) == labels[test_idx]
#     ).item() / len(test_idx)
#     print(
#         "Test Acc: {:.4f} | Test loss: {:.4f}".format(
#             test_acc, test_loss.item()
#         )
#     )
#     print()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="RGCN")
#     parser.add_argument(
#         "--dropout", type=float, default=0, help="dropout probability"
#     )
#     parser.add_argument(
#         "--n-hidden", type=int, default=16, help="number of hidden units"
#     )
#     parser.add_argument("--gpu", type=int, default=-1, help="gpu")
#     parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
#     parser.add_argument(
#         "--n-bases",
#         type=int,
#         default=-1,
#         help="number of filter weight matrices, default: -1 [use all]",
#     )
#     parser.add_argument(
#         "--n-layers", type=int, default=2, help="number of propagation rounds"
#     )
#     parser.add_argument(
#         "-e",
#         "--n-epochs",
#         type=int,
#         default=50,
#         help="number of training epochs",
#     )
#     parser.add_argument(
#         "-d", "--dataset", type=str, required=True, help="dataset to use"
#     )
#     parser.add_argument(
#         "--model_path", type=str, default=None, help="path for save the model"
#     )
#     parser.add_argument("--l2norm", type=float, default=0, help="l2 norm coef")
#     parser.add_argument(
#         "--use-self-loop",
#         default=False,
#         action="store_true",
#         help="include self feature as a special relation",
#     )
#     fp = parser.add_mutually_exclusive_group(required=False)
#     fp.add_argument("--validation", dest="validation", action="store_true")
#     fp.add_argument("--testing", dest="validation", action="store_false")
#     parser.set_defaults(validation=True)

#     args = parser.parse_args()
#     print(args)
#     main(args)
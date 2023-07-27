"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import time
import sys

import torch

sys.path.append("../../../")
sys.path.append("../")
from dgl.dataloading import MultiLayerNeighborSampler, NodeCollator


import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
import dgl.dataloading
import itertools
from MAGNN import ENCODERS, MAGNNMinibatch
from GNN.dgl_dataset import TwitterDataset
from GNN.models.utils import set_random_seed
from GNN.models.clustering.metrics import AverageNonzeroTripletsMetric
from GNN.models.clustering.loss import OnlineTripletLoss, HardestNegativeTripletSelector, RandomNegativeTripletSelector, ClusterLoss
from GNN.models.clustering.clusters import run_kmeans, run_kmeans_in_train, run_hdbscan, run_hdbscan_in_train, run_dbscan, run_dbscan_in_train
from GNN.models.clustering.MAGNN.metapath import to_ntype_list, metapath_based_graph


def load_subtensors(blocks, features):
    h_list = []
    for block in blocks:
        input_nodes = block.srcdata[dgl.NID]
        h_list.append(features[input_nodes])
    return h_list
def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb

def extract_graph_features(g, model, category, labels):
    with th.no_grad():
        model.eval()
        extract_nids = g.nodes[category]
        # print(extract_nids)
        extract_features = model()[category]
        extract_labels = labels  # labels of all nodes

    return extract_nids, extract_features, extract_labels

def evaluate(model, mgs, t_features, labels, val_nid, batch_size,
             epoch, is_validation, save_path, device="cpu"):
    model.eval()

    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'
    extract_features = []
    extract_labels = []

    sampler = MultiLayerNeighborSampler([100])
    collators = [NodeCollator(mg, val_nid, sampler) for mg in mgs]
    print(f'collators: {collators}')
    dataloader = DataLoader(val_nid, batch_size=args.batch_size, shuffle=True, drop_last=True)

    with th.no_grad():
        # for i, (input_nodes, seeds, blocks) in enumerate(loader):
        for i, batch in enumerate(dataloader):
            gs = [collator.collate(batch)[2][0] for collator in collators]
            # h_list = load_subtensors(gs, t_features)
            features = model(gs, t_features)
            batch_tic = time.time()
            lbl = labels[batch]
            extract_features.append(features.cpu().numpy())
            extract_labels.append(lbl)

    extract_features = th.Tensor(np.concatenate(extract_features))
    extract_labels = th.Tensor(np.concatenate(extract_labels))
    nids = th.Tensor(list(range(len(extract_labels)))).int()

    n_tweets, n_classes, nmi = run_kmeans(extract_features, extract_labels, nids)
    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode + ' NMI: '
    message += str(nmi)

    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
    print(message)

    return nmi

def train_process(args, split):
    if args.dataset == "aifb":
        dataset = AIFBDataset()
    elif args.dataset == "mutag":
        dataset = MUTAGDataset()
    elif args.dataset == "bgs":
        dataset = BGSDataset()
    elif args.dataset == "am":
        dataset = AMDataset()
    elif args.dataset == "twitter":
        dataset = TwitterDataset("../incremental_graph/", split=split)
    else:
        raise ValueError()

    g = dataset[0]
    metapath_list = [["t-u", "u-t"], ["t-w", "w-t"], ["t-h", "h-t"], ["t-e", "e-t"]]
    print(g.number_of_nodes(), g.number_of_edges())
    category = dataset.predict_category
    print(f'category: {category}')
    # metapaths = {'tweet': [["t-u", "u-t"], ["t-w", "w-t"], ["t-h", "h-t"], ["t-e", "e-t"]]}
    # print(g.number_of_nodes())
    if g.number_of_nodes() < 10:
        return
    num_classes = dataset.num_classes
    num_metapaths = len(metapath_list)

    train_mask = g.nodes[category].data.pop("train_mask")
    # test_mask = g.nodes[category].data.pop("test_mask")
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    # test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop("labels")
    t_features = g.nodes[category].data["features"].to(th.float32)
    save_path = os.path.join(args.save_path, "exp1")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[: len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5 :]
    else:
        val_idx = train_idx
    num_neighbors = args.num_neighbors

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        device = "cuda:%d" % args.gpu
        g = g.to("cuda:%d" % args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        val_idx = val_idx.cuda()
        t_features = t_features.cuda()
        # test_idx = test_idx.cuda()
        # Create PyTorch DataLoader for constructing blocks
    else:
        device = "cpu"

    # ---------------------------------------------------
    # create model
    # meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    g.nodes['user'].data['features'] = torch.unsqueeze(g.nodes['user'].data['features'], dim=1)
    in_dims={type: g.nodes[type].data['features'].shape for type in g.ntypes}
    # print(f"in_dims: {in_dims}")

    node_features = {ntype: g.nodes[ntype].data["features"].to(torch.float32) for ntype in g.ntypes}

    mgs = [metapath_based_graph(g, metapath) for metapath in metapath_list]
    mgs[0].ndata['feat'] = g.nodes['tweet'].data['features']
    metapaths_ntype = [to_ntype_list(g, metapath) for metapath in metapath_list]
    # sampler = MultiLayerNeighborSampler([args.fanout] * args.n_layers)
    sampler = MultiLayerNeighborSampler([10])
    collators = [NodeCollator(mg, train_idx, sampler) for mg in mgs]
    # print(f'collators: {collators}')
    loader = DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # loader = dgl.dataloading.DataLoader(
    #     g,
    #     {category: train_idx},
    #     sampler,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True
    # )

    model = MAGNNMinibatch(
        g,
        ntype=category,
        metapaths=metapaths_ntype,
        in_dims={type: g.nodes[type].data['features'].shape[1]
            # if type != 'user'
            # else g.nodes[type].data['features'].shape[0]
                 for type in g.ntypes},
        hidden_dim=args.n_hidden,
        embed_size=t_features.shape[1],
        out_dim=args.n_output,
        num_heads=args.n_heads,
        encoder=args.encoder,
        dropout=args.dropout,
    )
    # ---------------------------------------------------

    total_params = sum(p.numel() for p in model.parameters())
    print("total_params: {:d}".format(total_params))
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("total trainable params: {:d}".format(total_trainable_params))

    if not split==0:
        last_model_path = os.path.join(save_path, "models", str(split-1), "best.pt")
        model.load_state_dict(th.load(last_model_path), strict=True)

    if use_cuda:
        model.cuda()

    if args.use_hardest_neg:
        tr_loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
    else:
        tr_loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))
    kl_loss_fn = ClusterLoss(alpha=1)
    # optimizer
    all_params = model.parameters()
    optimizer = th.optim.Adam(
        all_params, lr=args.lr, weight_decay=args.l2norm
    )

    # Metrics
    # Counts average number of nonzero triplets found in minibatches
    metrics = [AverageNonzeroTripletsMetric()]

    # training loop
    print("start training...")
    model.train()
    best_vali_nmi = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_nmi = []

    # ---------------------------------------------------
    for epoch in range(args.n_epochs):
        # for i, (input_nodes, seeds, blocks) in enumerate(loader):
        for i, batch in enumerate(loader):
            gs = [collator.collate(batch)[2][0] for collator in collators]
            # print(f'gs from batch: {gs}')
            features = model(gs, node_features)
            # blocks = [block.to(device) for block in blocks]
            # seeds = seeds[category]  # we only predict the nodes with type "category"
            batch_tic = time.time()
            # lbl = labels[seeds]
            lbl = labels[batch]
            if use_cuda:
                lbl = lbl.cuda()
            # features = model(blocks)[category]
            # ---------------------------------------------------

            n_tweets, n_classes, centers, nmi = run_kmeans_in_train(features, lbl)
            # print(features.shape, lbl.shape)
            # print(lbl)
            tr_loss = tr_loss_fn(features, lbl)
            kl_loss = kl_loss_fn(features, centers)
            loss = tr_loss[0] + 10 * kl_loss
            for metric in metrics:
                metric(features, lbl, tr_loss)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            for metric in metrics:
                print('\t{}: {:.4f}'.format(metric.name(), metric.value()))
            if torch.is_tensor(loss):
                loss = loss.item()
            if torch.is_tensor(nmi):
                nmi = nmi.item()
            print(
                "Epoch {:05d} | Batch {:03d} | {}: {:.4f} | TR Loss: {:.4f} | KL Loss: {:.4f} | Train Loss: {:.4f} | Train NMI: {:.4f} | Time: {:.4f}".format(
                    epoch, i, metrics[0].name(), metrics[0].value(), tr_loss[0], kl_loss*10, loss, nmi, time.time() - batch_tic
                )
            )

        # ---------------------------------------------------
        val_nmi = evaluate(model, mgs, node_features, labels,
            val_idx, args.batch_size, epoch, is_validation=True,
            save_path=save_path, device=device)
        # ---------------------------------------------------

        all_vali_nmi.append(val_nmi)

        # Early stop
        if val_nmi > best_vali_nmi:
            best_vali_nmi = val_nmi
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = os.path.join(save_path, 'models', str(split))
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = os.path.join(model_path, 'best.pt')
            th.save(model.state_dict(), p)
            print('Best model saved after epoch ', str(epoch))
        else:
            wait += 1
        if wait == args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break

def infer_process(args, split):
    if args.dataset == "aifb":
        dataset = AIFBDataset()
    elif args.dataset == "mutag":
        dataset = MUTAGDataset()
    elif args.dataset == "bgs":
        dataset = BGSDataset()
    elif args.dataset == "am":
        dataset = AMDataset()
    elif args.dataset == "twitter":
        dataset = TwitterDataset("./incremental_graph/", split=split)
    else:
        raise ValueError()

    g = dataset[0]
    category = dataset.predict_category
    metapath_list = [["t-u", "u-t"], ["t-w", "w-t"], ["t-h", "h-t"], ["t-e", "e-t"]]
    if g.number_of_nodes() < 10:
        print(f"g.number_of_nodes is {g.number_of_nodes()} less then 10")
        return
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop("train_mask")
    # test_mask = g.nodes[category].data.pop("test_mask")
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = train_idx[:]
    # test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop("labels")
    t_features = g.nodes[category].data["features"].to(th.float32)
    save_path = os.path.join(args.save_path, "exp1")

    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[: len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5 :]
    else:
        val_idx = train_idx
    num_neighbors = args.num_neighbors
    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        device = "cuda:%d" % args.gpu
        g = g.to("cuda:%d" % args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        val_idx = val_idx.cuda()
        t_features = t_features.cuda()
    else:
        device = "cpu"

    # ---------------------------------------------------
    # create model
    # meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    g.nodes['user'].data['features'] = torch.unsqueeze(g.nodes['user'].data['features'], dim=1)
    node_features = {ntype: g.nodes[ntype].data["features"].to(torch.float32) for ntype in g.ntypes}
    mgs = [metapath_based_graph(g, metapath) for metapath in metapath_list]
    mgs[0].ndata['feat'] = g.nodes['tweet'].data['features']
    metapaths_ntype = [to_ntype_list(g, metapath) for metapath in metapath_list]

    model = MAGNNMinibatch(
        g,
        ntype=category,
        metapaths=metapaths_ntype,
        in_dims={type: g.nodes[type].data['features'].shape[1] for type in g.ntypes},
        hidden_dim=args.n_hidden,
        embed_size=t_features.shape[1],
        out_dim=args.n_output,
        num_heads=args.n_heads,
        encoder=args.encoder,
        dropout=args.dropout,
    )
    # ---------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    print("total_params: {:d}".format(total_params))
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("total trainable params: {:d}".format(total_trainable_params))

    if not split==0:
        last_model_path = os.path.join(save_path, "models", str(split-1), "best.pt")
        model.load_state_dict(th.load(last_model_path), strict=True)

    if use_cuda:
        model.cuda()

    # ---------------------------------------------------
    val_nmi = evaluate(model, mgs, node_features, labels,
                       test_idx, args.batch_size, -1, is_validation=True,
                       save_path=save_path, device=device)
    # ---------------------------------------------------


def main(args):
    # load graph data
    set_random_seed(args.seed)
    for i in range(0, 21):
        print("Training for Message {}".format(i))
        train_process(args, i)
        print("Testing for Message {}".format(i))
        infer_process(args, i+1)
        # exit(0)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--encoder', choices=list(ENCODERS.keys()), default='linear',
                        help='metapath instance encoder')
    parser.add_argument("--fanout", type=int, default=4, help="Fan-out of neighbor sampling.")
    parser.add_argument("--dropout", type=float, default=0, help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden units")
    parser.add_argument("--n-output", type=int, default=16, help="number of output feature dims")
    parser.add_argument("--n-embed", type=int, default=16, help="dim of embedding")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-heads", type=int, default=4, help="number of attention heads",)
    parser.add_argument("--n-layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, default="twitter", help="dataset to use"    )
    parser.add_argument("--l2norm", type=float, default=0, help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action="store_true", help="include self feature as a special relation",)
    parser.add_argument("--margin", type=float, default=3., help="Margin for triplet selection")
    parser.add_argument("--use_hardest_neg", default=True, action="store_true", help="whether to use hardest negative in triplet selection")
    parser.add_argument("--batch-size", type=int, default=100, help="Mini-batch size. If -1, use full graph training.",    )
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--save_path", type=str, default='.', help="Path to save model and results")
    parser.add_argument("--num_neighbors", type=int, default=20)
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument("--validation", dest="validation", action="store_true")
    fp.add_argument("--testing", dest="validation", action="store_false")

    parser.set_defaults(validation=True)

    args = parser.parse_args()
    main(args)

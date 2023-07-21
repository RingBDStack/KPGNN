"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import time
import sys
sys.path.insert(0, "../../")
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
import dgl.dataloading
from dgl_dataset import TwitterDataset
from RGCN import EntityCluster, RelGraphEmbed_Pretrain
from metrics import AverageNonzeroTripletsMetric
from loss import OnlineTripletLoss, HardestNegativeTripletSelector, RandomNegativeTripletSelector
import itertools
from clusters import run_kmeans

def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb

def extract_graph_features(g, model, category, labels):
    with th.no_grad():
        model.eval()
        extract_nids = g.nodes(category)
        extract_features = model()[category]
        extract_labels = labels  # labels of all nodes

    return extract_nids, extract_features, extract_labels

def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, is_validation=True):
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, nmi = run_kmeans(extract_features, extract_labels, indices)
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
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, nmi = run_kmeans(extract_features, extract_labels, indices,
                                              save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' NMI: '
        message += str(nmi)
    message += '\n'

    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
    print(message)

    return nmi

def train_process(args, split):
    device = "cpu"
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
    # print(g.number_of_nodes())
    if g.number_of_nodes() < 10:
        return
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop("train_mask")
    test_mask = g.nodes[category].data.pop("test_mask")
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop("labels")
    save_path = args.save_path

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

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [args.fanout] * args.n_layers
    )
    loader = dgl.dataloading.DataLoader(
        g,
        {category: train_idx},
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        g = g.to("cuda:%d" % args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()

    # create model
    model = EntityCluster(
        g,
        args.n_hidden,
        args.n_output,
        num_bases=args.n_bases,
        num_hidden_layers=args.n_layers - 2,
        dropout=args.dropout,
        use_self_loop=args.use_self_loop,
    )
    print(model)
    if not split==0:
        last_model_path = os.path.join(save_path, "models", str(split-1), "best.pt")
        model.load_state_dict(th.load(last_model_path), strict=True)

    if use_cuda:
        model.cuda()

    # loss
    if args.use_hardest_neg:
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))
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
    dur = []
    model.train()
    best_vali_nmi = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_nmi = []
    for epoch in range(args.n_epochs):
        for i, (input_nodes, seeds, blocks) in enumerate(loader):   
            # print("Input_nodes: {}, Seeds: {}, Blocks: {}".format(input_nodes, seeds, blocks))
            blocks = [blk.to(device) for blk in blocks]
            seeds = seeds[
                category
            ]  # we only predict the nodes with type "category"
            batch_tic = time.time()
            lbl = labels[seeds]
            if use_cuda:
                emb = {k: e.cuda() for k, e in emb.items()}
                lbl = lbl.cuda()
            features = model(blocks)[category]
            loss_outputs = loss_fn(features, lbl)
            loss = loss_outputs[0]
            for metric in metrics:
                metric(features, lbl, loss_outputs)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            for metric in metrics:
                print('\t{}: {:.4f}'.format(metric.name(), metric.value()))
            print(
                "Epoch {:05d} | Batch {:03d} | {}: {:.4f} | Train Loss: {:.4f} | Time: {:.4f}".format(
                    epoch, i, metrics[0].name(), metrics[0].value(), loss.item(), time.time() - batch_tic
                )
            )
        extract_nids, extract_features, extract_labels = extract_graph_features(g, model, category, labels)
        val_nmi = evaluate(extract_features, extract_labels, val_idx, epoch, 0, save_path, True)
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
    device = "cpu"
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
    # print(g.number_of_nodes())
    if g.number_of_nodes() < 10:
        return

    labels = g.nodes[category].data.pop("labels")
    save_path = args.save_path

    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [args.fanout] * args.n_layers
    )
    loader = dgl.dataloading.DataLoader(
        g,
        {category: list(range(len(labels)))},
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        g = g.to("cuda:%d" % args.gpu)
        labels = labels.cuda()

    # create model
    
    model = EntityCluster(
        g,
        args.n_hidden,
        args.n_output,
        num_bases=args.n_bases,
        num_hidden_layers=args.n_layers - 2,
        dropout=args.dropout,
        use_self_loop=args.use_self_loop,
    )
    if not split==0:
        last_model_path = os.path.join(save_path, "models", str(split-1), "best.pt")
        model.load_state_dict(th.load(last_model_path), strict=True)
    if use_cuda:
        model.cuda()

    # loss
    if args.use_hardest_neg:
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))
    # optimizer
    all_params = model.parameters()
    optimizer = th.optim.Adam(
        all_params, lr=args.lr, weight_decay=args.l2norm
    )

    # Metrics
    # Counts average number of nonzero triplets found in minibatches
    metrics = [AverageNonzeroTripletsMetric()]

    # training loop
    print("start testing...")
    dur = []
    model.eval()
    best_vali_nmi = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_nmi = []
    for epoch in range(1):
        for i, (input_nodes, seeds, blocks) in enumerate(loader):   
            # print("Input_nodes: {}, Seeds: {}, Blocks: {}".format(input_nodes, seeds, blocks))
            blocks = [blk.to(device) for blk in blocks]
            seeds = seeds[
                category
            ]  # we only predict the nodes with type "category"
            batch_tic = time.time()
            lbl = labels[seeds]
            if use_cuda:
                emb = {k: e.cuda() for k, e in emb.items()}
                lbl = lbl.cuda()
            features = model(blocks)[category]
            loss_outputs = loss_fn(features, lbl)
            loss = loss_outputs[0]
            for metric in metrics:
                metric(features, lbl, loss_outputs)

            # optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            # optimizer.step()
            for metric in metrics:
                print('\t{}: {:.4f}'.format(metric.name(), metric.value()))
            print(
                "Epoch {:05d} | Batch {:03d} | {}: {:.4f} | Train Loss: {:.4f} | Time: {:.4f}".format(
                    epoch, i, metrics[0].name(), metrics[0].value(), loss.item(), time.time() - batch_tic
                )
            )
        extract_nids, extract_features, extract_labels = extract_graph_features(g, model, category, labels)
        val_nmi = evaluate(extract_features, extract_labels, extract_nids, epoch, 0, save_path, True)
        all_vali_nmi.append(val_nmi)
    
def main(args):
    # load graph data
    for i in range(0, 22):
        train_process(args, i)
        if i > 0:
            infer_process(args, i)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument(
        "--dropout", type=float, default=0, help="dropout probability"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=16, help="number of hidden units"
    )
    parser.add_argument(
        "--n-output", type=int, default=16, help="number of output feature dims"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--n-bases",
        type=int,
        default=-1,
        help="number of filter weight matrices, default: -1 [use all]",
    )
    parser.add_argument(
        "--n-layers", type=int, default=2, help="number of propagation rounds"
    )
    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=50,
        help="number of training epochs",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="dataset to use"
    )
    parser.add_argument("--l2norm", type=float, default=0, help="l2 norm coef")
    parser.add_argument(
        "--use-self-loop",
        default=False,
        action="store_true",
        help="include self feature as a special relation",
    )
    parser.add_argument("--margin", type=float, default=3., help="Margin for triplet selection")
    parser.add_argument("--use_hardest_neg", default=True, action="store_true", help="whether to use hardest negative in triplet selection")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Mini-batch size. If -1, use full graph training.",
    )
    parser.add_argument(
        "--fanout", type=int, default=4, help="Fan-out of neighbor sampling."
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping."
    )
    parser.add_argument("--save_path", type=str, default=None, help="Path to save model and results")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument("--validation", dest="validation", action="store_true")
    fp.add_argument("--testing", dest="validation", action="store_false")
    
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    main(args)
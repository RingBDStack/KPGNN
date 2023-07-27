import argparse
import time
import sys
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl.dataloading
from datasets.dgl_dataset import TwitterDataset
from models.HAN import HAN, HANSampler
from utils.metrics import AverageNonzeroTripletsMetric
from utils.loss import OnlineTripletLoss, HardestNegativeTripletSelector, RandomNegativeTripletSelector, ClusterLoss
import itertools
from utils.clusters import run_kmeans, run_kmeans_in_train, run_hdbscan, run_hdbscan_in_train, run_dbscan, run_dbscan_in_train
from torch.utils.data import DataLoader

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

def evaluate(args, model,
    g,
    t_features,
    labels,
    val_nid,
    batch_size,
    epoch,
    is_validation,
    save_path,
    device="cpu"):
    model.eval()
    loader = construct_loader(args, g, device, val_nid)
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'
    extract_features = []
    extract_labels = []
    with th.no_grad():
        for step, (seeds, blocks) in enumerate(loader):
            h_list = load_subtensors(blocks, t_features) # 获取到对应的初始feature
            blocks = [block for block in blocks]
            hs = [h for h in h_list]
            blocks = [block.to(device) for block in blocks]
            hs = [h.to(device) for h in h_list]

            features = model(blocks, hs)
            labels_batch = labels[np.asarray(seeds)].cpu().numpy()

            extract_features.append(features.cpu().numpy())
            extract_labels.append(labels_batch)
    
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

def construct_model(args, graph, init_dim=0):
    model = None
    model_type = args.model
    if model_type == "HAN":
        if init_dim == 0:
            raise ValueError("Init-dim of target nodes must be specified when using model HAN. ")
        metapath_list = [["t-u", "u-t"], ["t-w", "w-t"], ["t-h", "h-t"], ["t-e", "e-t"]]
        model = HAN(
            graph,
            num_metapath=len(metapath_list),
            embed_size=init_dim,
            hidden_size=args.n_hidden,
            out_size=args.n_output,
            num_heads=args.n_heads,
            dropout=args.dropout,
            num_layer=args.n_layers
        )
    else:
        raise ValueError("Unimplemented model.")
    return model

def construct_loader(args, g, device, seeds):
    loader = None
    sampler_type = args.sampler
    if sampler_type == "RandomWalk":
        metapath_list = [["t-u", "u-t"], ["t-w", "w-t"], ["t-h", "h-t"], ["t-e", "e-t"]]
        num_neighbors = args.num_neighbors
        han_sampler = HANSampler(g, metapath_list, num_neighbors, device=device)
        loader = DataLoader(
            dataset=seeds.cpu().numpy(),
            batch_size=args.batch_size,
            collate_fn=han_sampler.sample_blocks,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
    elif sampler_type == "MNeighbor":
        pass
    else:
        raise ValueError("Unimplemented dataloader.")
    return loader


def train_process(args, split):
    # Construct dataset
    dataset = TwitterDataset(args.graph_path, split=split, raw_dir=args.data_path)

    g = dataset[0]
    print(g.number_of_nodes(), g.number_of_edges())
    if g.number_of_nodes() < 10:
        return

    
    # Organize data from graph to node vectors
    category = dataset.predict_category
    train_mask = g.nodes[category].data.pop("train_mask")
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop("labels")
    t_features = g.nodes[category].data["features"].to(th.float32)
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[: len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5 :]
    else:
        val_idx = train_idx

    # minibatch sampler and data loader
    device = "cpu"
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        device = "cuda:%d" % args.gpu

    loader = construct_loader(args, g, device, train_idx)
    
    # create model
    # meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    model = construct_model(args, g, init_dim=t_features.shape[1])

    # check cuda
    g = g.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    t_features = t_features.to(device)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("total_params: {:d}".format(total_params))
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("total trainable params: {:d}".format(total_trainable_params))

    # Load Parameters in last message block
    if not split==0:
        last_model_path = os.path.join(save_path, "models", str(split-1), "best.pt")
        model.load_state_dict(th.load(last_model_path), strict=True)

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
    for epoch in range(args.n_epochs):
        for i, (seeds, blocks) in enumerate(loader):   
            # print("Seeds: {}, Blocks: {}".format(len(seeds), blocks))
            h_list = load_subtensors(blocks, t_features)
            blocks = [block.to(device) for block in blocks]
            hs = [h.to(device) for h in h_list]
            batch_tic = time.time()
            lbl = labels[seeds]
            if use_cuda:
                lbl = lbl.cuda()
            features = model(blocks, hs)
            # n_tweets, n_classes, centers, nmi = run_kmeans_in_train(features, lbl)
            n_tweets, n_classes, centers, nmi = run_kmeans_in_train(features, lbl)
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
            print(
                "Epoch {:05d} | Batch {:03d} | {}: {:.4f} | TR Loss: {:.4f} | KL Loss: {:.4f} | Train Loss: {:.4f} | Train NMI: {:.4f} | Time: {:.4f}".format(
                    epoch, i, metrics[0].name(), metrics[0].value(), tr_loss[0], kl_loss*10, loss.item(), nmi.item(), time.time() - batch_tic
                )
            )
        
        val_nmi = evaluate(
            args, 
            model,
            g,
            t_features,
            labels,
            val_idx,
            args.batch_size,
            epoch, 
            is_validation=True,
            save_path=save_path,
            device=device)
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
    dataset = TwitterDataset("./datasets/incremental_graph/", split=split)

    g = dataset[0]
    category = dataset.predict_category
    metapath_list = [["t-u", "u-t"], ["t-w", "w-t"], ["t-h", "h-t"], ["t-e", "e-t"]]
    # print(g.number_of_nodes())
    if g.number_of_nodes() < 10:
        return
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop("train_mask")
    # test_mask = g.nodes[category].data.pop("test_mask")
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = train_idx[:]
    # test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop("labels")
    t_features = g.nodes[category].data["features"].to(th.float32)
    save_path = os.path.join(args.save_path, "HAN")

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
    # create model
    # meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    model = HAN(
        g,
        num_metapath=len(metapath_list),
        embed_size=t_features.shape[1],
        hidden_size=args.n_hidden,
        out_size=args.n_output,
        num_heads=args.n_heads,
        dropout=args.dropout,
    )
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

    val_nmi = evaluate(
        model,
        g,
        metapath_list,
        num_neighbors,
        t_features,
        labels,
        test_idx,
        args.batch_size,
        -1, 
        is_validation=False,
        save_path=save_path,
        device=device)

    
def main(args):
    # load graph data
    for i in range(0, 21):
        print("Training for Message {}".format(i))
        train_process(args, i)
        print("Testing for Message {}".format(i))
        infer_process(args, i+1)
    
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POISED")
    # 数据相关
    parser.add_argument("--graph-path", type=str, default='./datasets/incremental_graph/', help="Path to save and load constructed graph")
    parser.add_argument("--data-path", type=str, default='./datasets/Twitter/', help="Path to load raw data")
    parser.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 0.') #还未实现
    
    
    # 记录/输出相关
    parser.add_argument("--save_path", type=str, default='./experiments/exp', help="Path to save model and results")
    
    # 模型结构相关
    parser.add_argument("--model", type=str, default="MAGNN", help="model to use")
    parser.add_argument("--n-layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--n-hidden", type=int, default=64, help="number of hidden units")
    parser.add_argument("--n-output", type=int, default=16, help="number of output feature dims")
    parser.add_argument("--n-embed", type=int, default=16, help="dim of embedding")
    parser.add_argument("--n-heads", type=int, default=8, help="number of attention heads",)
    parser.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    parser.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    
    # 训练相关
    parser.add_argument("--dropout", type=float, default=0, help="dropout probability")
    parser.add_argument("-e", "--n-epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--batch-size", type=int, default=100, help="Mini-batch size. If -1, use full graph training.",    )
    parser.add_argument("--num_neighbors", type=int, default=20, help="Neighbor numbers for random walk sampler. Only for HAN now.")
    parser.add_argument("--sampler", type=str, default="RandomWalk", help="sampler for minibatch, RandomWalk or MNeighbor")
    parser.add_argument("--l2norm", type=float, default=0, help="weight decay for optimizer")
    # 损失函数相关
    parser.add_argument("--margin", type=float, default=3., help="Margin for triplet selection")
    parser.add_argument("--use_hardest_neg", default=True, action="store_true", help="whether to use hardest negative in triplet selection")
    
    # 增量聚类相关
    parser.add_argument("--eps", type=float, default=0.5, help="Eps for incremental dbscan")
    parser.add_argument("--n-prototype", type=int, default=100, help="Prototype numbers for n-prototype")

    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument("--validation", dest="validation", action="store_true")
    fp.add_argument("--testing", dest="validation", action="store_false")
    
    parser.set_defaults(validation=False)

    args = parser.parse_args()
    main(args)
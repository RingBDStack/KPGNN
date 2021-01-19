
import numpy as np
import json
import argparse
from torch.utils.data import Dataset
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
from metrics import AverageNonzeroTripletsMetric
import time
from time import localtime, strftime
import os
import pickle
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn import metrics


# Dataset
class SocialDataset(Dataset):
    def __init__(self, path, index):
        self.features = np.load(path + '/' + str(index) + '/features.npy')
        temp =  np.load(path + '/' + str(index) + '/labels.npy', allow_pickle=True)
        self.labels = np.asarray([int(each) for each in temp])
        self.matrix = self.load_adj_matrix(path, index)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def load_adj_matrix(self, path, index):
        s_bool_A_tid_tid = sparse.load_npz(path + '/' + str(index) + '/s_bool_A_tid_tid.npz')
        print("Sparse binary adjacency matrix loaded.")
        return s_bool_A_tid_tid

    # Used by remove_obsolete mode 1  
    def remove_obsolete_nodes(self, indices_to_remove = None): # indices_to_remove: list
        #torch.range(0, (self.labels.shape[0] - 1), dtype=torch.long)
        if indices_to_remove is not None:
            all_indices = np.arange(0, self.labels.shape[0]).tolist()
            indices_to_keep = list(set(all_indices) - set(indices_to_remove))
            self.features = self.features[indices_to_keep, :]
            self.labels = self.labels[indices_to_keep]
            self.matrix = self.matrix[indices_to_keep, :]
            self.matrix = self.matrix[:, indices_to_keep]

def graph_statistics(G, save_path):
    message = '\nGraph statistics:\n'
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges/2) // num_nodes
    in_degrees = G.in_degrees()
    isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = (in_degrees == isolated_nodes)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt') 
    num_isolated_nodes = torch.sum(isolated_nodes).item()

    message += 'We have ' + str(num_nodes) + ' nodes.\n'
    message += 'We have ' + str(num_edges/2) + ' in-edges.\n'
    message += 'Average degree: ' + str(ave_degree) + '\n'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "a") as f:
        f.write(message)
    
    return num_isolated_nodes

# This function generates train and validation indices for initial/maintenance epochs and test indices for inference(prediction) epochs
# If remove_obsolete mode 0 or 1:
    # For initial/maintenance epochs:
    #  - The first (train_i + 1) blocks (blocks 0, ..., train_i) are used as training set (with explicit labels)
    #  - Randomly sample validation_percent of the training indices as validation indices
    # For inference(prediction) epochs:
    #  - The (i + 1)th block (block i) is used as test set
    # Note that other blocks (block train_i + 1, ..., i - 1) are also in the graph (without explicit labels, 
    # only their features and structural info are leveraged)
# If remove_obsolete mode 2:
    # For initial/maintenance epochs:
    #  - The (i + 1) = (train_i + 1)th block (block train_i = i) is used as training set (with explicit labels)
    #  - Randomly sample validation_percent of the training indices as validation indices
    # For inference(prediction) epochs:
    #  - The (i + 1)th block (block i) is used as test set
def generateMasks(length, data_split, train_i, i, validation_percent = 0.2, save_path = None, num_indices_to_remove = 0):
    if args.remove_obsolete == 0 or args.remove_obsolete == 1: #remove_obsolete mode 0 or 1
        # verify total number of nodes
        assert length == (np.sum(data_split[:i+1]) - num_indices_to_remove)

        # If is in initial/maintenance epochs, generate train and validation indices
        if train_i == i:
            # randomly suffle the training indices
            train_length = np.sum(data_split[:train_i+1])
            train_length -= num_indices_to_remove
            train_indices = torch.randperm(int(train_length))
            # get total number of validation indices
            n_validation_samples = int(train_length * validation_percent)
            # sample n_validation_samples validation indices and use the rest as training indices
            validation_indices = train_indices[:n_validation_samples]
            train_indices = train_indices[n_validation_samples:]
            if save_path is not None:
                torch.save(validation_indices, save_path + '/validation_indices.pt') 
                torch.save(train_indices, save_path + '/train_indices.pt')
                validation_indices = torch.load(save_path + '/validation_indices.pt')
                train_indices = torch.load(save_path + '/train_indices.pt')
            return train_indices, validation_indices
        # If is in inference(prediction) epochs, generate test indices
        else:
            test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
            test_indices += (np.sum(data_split[:i]) - num_indices_to_remove)
            if save_path is not None:
                torch.save(test_indices, save_path + '/test_indices.pt') 
                test_indices = torch.load(save_path + '/test_indices.pt')
            return test_indices

    else: #remove_obsolete mode 2
        # verify total number of nodes
        assert length == data_split[i]

        # If is in initial/maintenance epochs, generate train and validation indices
        if train_i == i:
            # randomly suffle the graph indices
            train_indices = torch.randperm(length)
            # get total number of validation indices
            n_validation_samples = int(length * validation_percent)
            # sample n_validation_samples validation indices and use the rest as training indices
            validation_indices = train_indices[:n_validation_samples]
            train_indices = train_indices[n_validation_samples:]
            if save_path is not None:
                torch.save(validation_indices, save_path + '/validation_indices.pt') 
                torch.save(train_indices, save_path + '/train_indices.pt')
                validation_indices = torch.load(save_path + '/validation_indices.pt')
                train_indices = torch.load(save_path + '/train_indices.pt')
            return train_indices, validation_indices
        # If is in inference(prediction) epochs, generate test indices
        else:
            test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
            if save_path is not None:
                torch.save(test_indices, save_path + '/test_indices.pt') 
                test_indices = torch.load(save_path + '/test_indices.pt')
            return test_indices


# Utility function, finds the indices of the values' elements in tensor
def find(tensor, values):
    return torch.nonzero(tensor.cpu()[..., None] == values.cpu())

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual = False):
        super(GATLayer, self).__init__()
        # equation (1) reference: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, nf, layer_id):
        h = nf.layers[layer_id].data['h']
        # equation (1)
        z = self.fc(h)
        nf.layers[layer_id].data['z'] = z
        #print("test test test")
        A = nf.layer_parent_nid(layer_id)
        #print(A)
        #print(A.shape)
        A = A.unsqueeze(-1)
        B = nf.layer_parent_nid(layer_id+1)
        #print(B)
        #print(B.shape)
        B = B.unsqueeze(0)
       
        _, indices = torch.topk((A==B).int(), 1, 0)
        #print(indices)
        #print(indices.shape)
        #indices = np.asarray(indices)
        indices = indices.cpu().data.numpy()
        
        nf.layers[layer_id+1].data['z'] = z[indices]
        #print(nf.layers[layer_id+1].data['z'].shape)
        # equation (2)
        nf.apply_block(layer_id, self.edge_attention)
        # equation (3) & (4)
        nf.block_compute(layer_id, # block_id – The block to run the computation.
                         self.message_func, # Message function on the edges.
                         self.reduce_func) # Reduce function on the node.
        
        nf.layers[layer_id].data.pop('z')
        nf.layers[layer_id+1].data.pop('z')

        if self.use_residual:
            return z[indices] + nf.layers[layer_id + 1].data['h'] # residual connection
        return nf.layers[layer_id + 1].data['h']

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual = False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual))
        self.merge = merge

    def forward(self, nf, layer_id):
        head_outs = [attn_head(nf, layer_id) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual = False):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)

    def forward(self, nf, corrupt = False):
        features = nf.layers[0].data['features']
        if corrupt:
            nf.layers[0].data['h'] = features[torch.randperm(features.size()[0])]
        else:
            nf.layers[0].data['h'] = features
        h = self.layer1(nf, 0)
        h = F.elu(h)
        #print(h.shape)
        nf.layers[1].data['h'] = h
        h = self.layer2(nf, 1)
        return h

# Applies an average on seq, of shape (nodes, features)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)
        
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)
        #print("testing, shape of logits: ", logits.size())
        return logits

class DGI(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual = False):
        super(DGI, self).__init__()
        self.gat = GAT(in_dim, hidden_dim, out_dim, num_heads, use_residual)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid() 
        self.disc = Discriminator(out_dim)

    def forward(self, nf):
        h_1 = self.gat(nf, False)
        c = self.read(h_1)
        c = self.sigm(c)
        h_2 = self.gat(nf, True)
        ret = self.disc(c, h_1, h_2)
        return h_1, ret

    # Detach the return variables
    def embed(self, nf):
        h_1 = self.gat(nf, False)
        return h_1.detach()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)

def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)

# Compute the representations of all the nodes in g using model
def extract_embeddings(g, model, num_all_samples, labels):
    with torch.no_grad():
        model.eval()
        for batch_id, nf in enumerate(dgl.contrib.sampling.NeighborSampler(g, # sample from the whole graph (contain unseen nodes)
                                                    num_all_samples, # set batch size = the total number of nodes
                                                    1000, # set the expand_factor (the number of neighbors sampled from the neighbor list of a vertex) to None: get error: non-int expand_factor not supported
                                                    neighbor_type='in',
                                                    shuffle=False,
                                                    num_workers=32,
                                                    num_hops=2)):
            nf.copy_from_parent()
            if args.use_dgi:
                extract_features, _ = model(nf) # representations of all nodes
            else:
                extract_features = model(nf) # representations of all nodes
            extract_nids = nf.layer_parent_nid(-1).to(device=extract_features.device, dtype=torch.long) # node ids
            extract_labels = labels[extract_nids] # labels of all nodes
        assert batch_id == 0
        extract_nids = extract_nids.data.cpu().numpy()
        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()
        # generate train/test mask
        A = np.arange(num_all_samples)
        #print("A", A)
        assert (A==extract_nids).all()
        
    return (extract_nids, extract_features, extract_labels)

def save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, path, counter):
    np.savetxt(path + '/features_' + str(counter) + '.tsv', extract_features, delimiter = '\t')
    np.savetxt(path + '/labels_' + str(counter) + '.tsv', extract_labels, fmt='%i', delimiter = '\t')
    with open(path + '/labels_tags_' + str(counter) + '.tsv', 'w') as f:
        f.write('label\tmessage_id\ttrain_tag\n')
        for (label, mid, train_tag) in zip(extract_labels, extract_nids, extract_train_tags):
            f.write("%s\t%s\t%s\n" % (label, mid, train_tag))
    print("Embeddings after inference epoch " + str(counter) + " saved.")
    print()

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


def run_kmeans(extract_features, extract_labels, indices, isoPath = None): 
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)

    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, nmi)

def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, is_validation = True):
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
        n_tweets, n_classes, nmi = run_kmeans(extract_features, extract_labels, indices, save_path + '/isolated_nodes.pt')
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

# Inference(prediction)
def infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, train_indices = None, model = None, loss_fn_dgi = None, indices_to_remove = []):

    # make dir for graph i
    save_path_i = embedding_save_path + '/block_' + str(i)
    if not os.path.isdir(save_path_i):
        os.mkdir(save_path_i)

    # load data
    data = SocialDataset(args.data_path, i)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    in_feats = features.shape[1] # feature dimension

    # Construct graph
    g = dgl.DGLGraph(data.matrix) # graph that contains message blocks 0, ..., i if remove_obsolete = 0 or 1; graph that only contains message block i if remove_obsolete = 2
    num_isolated_nodes = graph_statistics(g, save_path_i)

    # if remove_obsolete is mode 1, resume or use the passed indices_to_remove to remove obsolete nodes from the graph
    if args.remove_obsolete == 1:

        if ((args.resume_path is not None) and (not args.resume_current) and (i == args.resume_point + 1) and (i > args.window_size)) \
            or (indices_to_remove == [] and i > args.window_size): # Resume indices_to_remove from the last maintain block

            temp_i = max(((i - 1) // args.window_size) * args.window_size, 0)
            indices_to_remove = np.load(embedding_save_path + '/block_' + str(temp_i) + '/indices_to_remove.npy').tolist()

        if indices_to_remove != []:

            # remove obsolete nodes from the graph
            data.remove_obsolete_nodes(indices_to_remove)
            features = torch.FloatTensor(data.features)
            labels = torch.LongTensor(data.labels)
            # Reconstruct graph
            g = dgl.DGLGraph(data.matrix) # graph that contains tweet blocks 0, ..., i
            num_isolated_nodes = graph_statistics(g, save_path_i)

    # generate or load test mask
    if args.mask_path is None:
        mask_path = save_path_i + '/masks'
        if not os.path.isdir(mask_path):
            os.mkdir(mask_path)
        test_indices = generateMasks(len(labels), data_split, train_i, i, args.validation_percent, mask_path, len(indices_to_remove))
    else:
        test_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/test_indices.pt')

    # Suppress warning
    g.set_n_initializer(dgl.init.zero_initializer)
    g.readonly()

    if args.use_cuda:
        features, labels = features.cuda(), labels.cuda()
        test_indices = test_indices.cuda()
        
    g.ndata['features'] = features

    if (args.resume_path is not None) and (not args.resume_current) and (i == args.resume_point + 1): # Resume model from the previous block and train_indices from the last initil/maintain block
        
        # Declare model
        if args.use_dgi:
            model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        else:
            model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

        if args.use_cuda:
            model.cuda()

        # Load model from resume_point
        model_path = embedding_save_path + '/block_' + str(args.resume_point) + '/models/best.pt'
        model.load_state_dict(torch.load(model_path))
        print("Resumed model from the previous block.")

        # Use resume_path as a flag
        args.resume_path = None

    if train_indices is None: # train_indices is used for continue training then predict
        if args.remove_obsolete == 0 or args.remove_obsolete == 1:
            # Resume train_indices from the last initil/maintain block
            temp_i = max(((i - 1) // args.window_size) * args.window_size, 0)
            train_indices = torch.load(embedding_save_path + '/block_' + str(temp_i) + '/masks/train_indices.pt')
        else:
            if args.n_infer_epochs != 0:
                print("==================================\n'continue training then predict' is unimplemented under remove_obsolete mode 2, will skip infer epochs.\n===================================\n")
                args.n_infer_epochs = 0

    # record test nmi of all epochs
    all_test_nmi = []
    # record the time spent in seconds on direct prediction
    time_predict = []

    # Directly predict
    message = "\n------------ Directly predict on block " + str(i) + " ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    start = time.time()
    # Infer the representations of all tweets
    extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
    # Predict: conduct kMeans clustering on the test(newly inserted) nodes and report NMI
    test_nmi = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes, save_path_i, False)
    seconds_spent = time.time() - start
    message = '\nDirect prediction took {:.2f} seconds'.format(seconds_spent)
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    all_test_nmi.append(test_nmi)
    time_predict.append(seconds_spent)
    np.save(save_path_i + '/time_predict.npy', np.asarray(time_predict))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-4)

    # Continue training then predict (leverage the features and structural info of the newly inserted block) 
    if args.n_infer_epochs != 0:
        message = "\n------------ Continue training then predict on block " + str(i) + " ------------\n"
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
    # record the time spent in seconds on each batch of all infer epochs
    seconds_infer_batches = []
    # record the time spent in mins on each epoch
    mins_infer_epochs = []
    for epoch in range(args.n_infer_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        if args.use_dgi:
            losses_triplet = []
            losses_dgi = []
        for metric in metrics:
            metric.reset()
        for batch_id, nf in enumerate(dgl.contrib.sampling.NeighborSampler(g, 
                                                        args.batch_size,
                                                        args.n_neighbors,
                                                        neighbor_type='in',
                                                        shuffle=True,
                                                        num_workers=32,
                                                        num_hops=2,
                                                        seed_nodes=train_indices)):
            start_batch = time.time()                                            
            nf.copy_from_parent()
            model.train()
            # forward
            if args.use_dgi:
                pred, ret = model(nf) # pred: representations of the sampled nodes (in the last layer of the NodeFlow), ret: discriminator results
            else:
                pred = model(nf) # Representations of the sampled nodes (in the last layer of the NodeFlow).
            batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
            batch_labels = labels[batch_nids]
            loss_outputs = loss_fn(pred, batch_labels)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            if args.use_dgi:
                n_samples = nf.layer_nid(-1).size()[0]
                lbl_1 = torch.ones(n_samples)
                lbl_2 = torch.zeros(n_samples)
                lbl = torch.cat((lbl_1, lbl_2), 0)
                if args.use_cuda:
                    lbl = lbl.cuda()
                losses_triplet.append(loss.item())
                loss_dgi = loss_fn_dgi(ret, lbl)
                losses_dgi.append(loss_dgi.item())
                loss += loss_dgi
                losses.append(loss.item())
            else:
                losses.append(loss.item())
            total_loss += loss.item()
            
            for metric in metrics:
                metric(pred, batch_labels, loss_outputs)
                
            if batch_id % args.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_id * args.batch_size, train_indices.shape[0],
                    100. * batch_id / (train_indices.shape[0] // args.batch_size), np.mean(losses))
                if args.use_dgi:
                    message += '\tLoss_triplet: {:.6f}'.format(np.mean(losses_triplet))
                    message += '\tLoss_dgi: {:.6f}'.format(np.mean(losses_dgi))
                for metric in metrics:
                    message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                print(message)
                with open(save_path_i + '/log.txt', 'a') as f:
                    f.write(message)
                losses = []

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_seconds_spent = time.time() - start_batch
            seconds_infer_batches.append(batch_seconds_spent)
            # end one batch
            
        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_infer_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch)/60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_infer_epochs.append(mins_spent)
        
        # Validation
        # Infer the representations of all tweets
        extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
        # Save the representations of all tweets
        #save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        # Evaluate the model: conduct kMeans clustering on the validation and report NMI
        test_nmi = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes, save_path_i, False)
        all_test_nmi.append(test_nmi)
        # end one epoch
    
    
    # Save model (fine-tuned from the above continue training process)
    model_path = save_path_i + '/models'
    os.mkdir(model_path)
    p = model_path + '/best.pt'
    torch.save(model.state_dict(), p)
    print('Model saved.')
    
    # Save all test nmi
    np.save(save_path_i + '/all_test_nmi.npy', np.asarray(all_test_nmi))
    print('Saved all test nmi.')
    # Save time spent on epochs
    np.save(save_path_i + '/mins_infer_epochs.npy', np.asarray(mins_infer_epochs))
    print('Saved mins_infer_epochs.')
    # Save time spent on batches
    np.save(save_path_i + '/seconds_infer_batches.npy', np.asarray(seconds_infer_batches))
    print('Saved seconds_infer_batches.')

    return model

# Train on initial/maintenance graphs
def initial_maintain(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model = None, loss_fn_dgi = None):

    # make dir for graph i
    save_path_i = embedding_save_path + '/block_' + str(i)
    if not os.path.isdir(save_path_i):
        os.mkdir(save_path_i)

    # load data
    data = SocialDataset(args.data_path, i)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    in_feats = features.shape[1] # feature dimension

    # Construct graph
    g = dgl.DGLGraph(data.matrix) # graph that contains message blocks 0, ..., i if remove_obsolete = 0 or 1; graph that only contains message block i if remove_obsolete = 2
    num_isolated_nodes = graph_statistics(g, save_path_i)

    # if remove_obsolete is mode 1, resume or generate indices_to_remove, then remove obsolete nodes from the graph
    if args.remove_obsolete == 1: 

        if (args.resume_path is not None) and args.resume_current and (i == args.resume_point) and (i != 0): # Resume indices_to_remove from the current block
            indices_to_remove = np.load(save_path_i + '/indices_to_remove.npy').tolist()

        elif i == 0: # generate empty indices_to_remove for initial block
            indices_to_remove = []
            # save indices_to_remove
            np.save(save_path_i + '/indices_to_remove.npy', np.asarray(indices_to_remove))

        else: # generate indices_to_remove for mantenance block
            # get the indices of all training nodes
            num_all_train_nodes = np.sum(data_split[:i+1])
            all_train_indices = np.arange(0, num_all_train_nodes).tolist()
            # get the number of old training nodes added before this maintenance
            num_old_train_nodes = np.sum(data_split[:i+1-args.window_size]) 
            # indices_to_keep: indices of nodes that are connected to the new training nodes added at this maintenance 
            # (include the indices of the new training nodes)
            indices_to_keep = list(set(data.matrix.indices[data.matrix.indptr[num_old_train_nodes]:]))
            # indices_to_remove is the difference between the indices of all training nodes and indices_to_keep
            indices_to_remove = list(set(all_train_indices) - set(indices_to_keep))
            # save indices_to_remove
            np.save(save_path_i + '/indices_to_remove.npy', np.asarray(indices_to_remove))

        if indices_to_remove != []:
            # remove obsolete nodes from the graph
            data.remove_obsolete_nodes(indices_to_remove)
            features = torch.FloatTensor(data.features)
            labels = torch.LongTensor(data.labels)
            # Reconstruct graph
            g = dgl.DGLGraph(data.matrix) # graph that contains tweet blocks 0, ..., i
            num_isolated_nodes = graph_statistics(g, save_path_i)

    else:

        indices_to_remove = []

    # generate or load training/validate/test masks
    if (args.resume_path is not None) and args.resume_current and (i == args.resume_point): # Resume masks from the current block

        train_indices = torch.load(save_path_i + '/masks/train_indices.pt')
        validation_indices = torch.load(save_path_i + '/masks/validation_indices.pt')
    if args.mask_path is None:

        mask_path = save_path_i + '/masks'
        if not os.path.isdir(mask_path):
            os.mkdir(mask_path)
        train_indices, validation_indices = generateMasks(len(labels), data_split, train_i, i, args.validation_percent, mask_path, len(indices_to_remove))
    
    else:
        train_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/train_indices.pt')
        validation_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/validation_indices.pt')

    # Suppress warning
    g.set_n_initializer(dgl.init.zero_initializer)
    g.readonly()

    if args.use_cuda:
        features, labels = features.cuda(), labels.cuda()
        train_indices, validation_indices = train_indices.cuda(), validation_indices.cuda()
        
    g.ndata['features'] = features

    if (args.resume_path is not None) and args.resume_current and (i == args.resume_point): # Resume model from the current block

        # Declare model
        if args.use_dgi:
            model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        else:
            model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

        if args.use_cuda:
            model.cuda()

        # Load model from resume_point
        model_path = embedding_save_path + '/block_' + str(args.resume_point) + '/models/best.pt'
        model.load_state_dict(torch.load(model_path))
        print("Resumed model from the current block.")

        # Use resume_path as a flag
        args.resume_path = None

    elif model is None: # Construct the initial model
        # Declare model
        if args.use_dgi:
            model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        else:
            model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

        if args.use_cuda:
            model.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-4)

    # Start training 
    message = "\n------------ Start initial training / maintaining using blocks 0 to " + str(i) + " ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    # record the highest validation nmi ever got for early stopping
    best_vali_nmi = 1e-9
    best_epoch = 0
    wait = 0 
    # record validation nmi of all epochs before early stop
    all_vali_nmi = []
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        if args.use_dgi:
            losses_triplet = []
            losses_dgi = []
        for metric in metrics:
            metric.reset()
        for batch_id, nf in enumerate(dgl.contrib.sampling.NeighborSampler(g, 
                                                        args.batch_size,
                                                        args.n_neighbors,
                                                        neighbor_type='in',
                                                        shuffle=True,
                                                        num_workers=32,
                                                        num_hops=2,
                                                        seed_nodes=train_indices)):
            start_batch = time.time()
            nf.copy_from_parent()
            model.train()
            # forward
            if args.use_dgi:
                pred, ret = model(nf) # pred: representations of the sampled nodes (in the last layer of the NodeFlow), ret: discriminator results
            else:
                pred = model(nf) # Representations of the sampled nodes (in the last layer of the NodeFlow).
            batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
            batch_labels = labels[batch_nids]
            loss_outputs = loss_fn(pred, batch_labels)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            if args.use_dgi:
                n_samples = nf.layer_nid(-1).size()[0]
                lbl_1 = torch.ones(n_samples)
                lbl_2 = torch.zeros(n_samples)
                lbl = torch.cat((lbl_1, lbl_2), 0)
                if args.use_cuda:
                    lbl = lbl.cuda()
                losses_triplet.append(loss.item())
                loss_dgi = loss_fn_dgi(ret, lbl)
                losses_dgi.append(loss_dgi.item())
                loss += loss_dgi
                losses.append(loss.item())
            else:
                losses.append(loss.item())
            total_loss += loss.item()
            
            for metric in metrics:
                metric(pred, batch_labels, loss_outputs)
                
            if batch_id % args.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_id * args.batch_size, train_indices.shape[0],
                    100. * batch_id / ((train_indices.shape[0] // args.batch_size) + 1), np.mean(losses))
                if args.use_dgi:
                    message += '\tLoss_triplet: {:.6f}'.format(np.mean(losses_triplet))
                    message += '\tLoss_dgi: {:.6f}'.format(np.mean(losses_dgi))
                for metric in metrics:
                    message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                print(message)
                with open(save_path_i + '/log.txt', 'a') as f:
                    f.write(message)
                losses = []

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch
            
        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch)/60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        # Validation
        # Infer the representations of all tweets
        extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
        # Save the representations of all tweets
        #save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        # Evaluate the model: conduct kMeans clustering on the validation and report NMI
        validation_nmi = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes, save_path_i, True)
        all_vali_nmi.append(validation_nmi)

        # Early stop
        if validation_nmi > best_vali_nmi:
            best_vali_nmi = validation_nmi
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)
            print('Best model saved after epoch ', str(epoch))
        else:
            wait += 1
        if wait == args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # Save all validation nmi
    np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
    # Save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # Save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')

    # Load the best model of the current block
    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print("Best model loaded.")

    if args.remove_obsolete == 2:
        return None, indices_to_remove, model
    return train_indices, indices_to_remove, model


def main(args):
    
    use_cuda = args.use_cuda and torch.cuda.is_available()
    print("Using CUDA:", use_cuda)

    # make dirs and save args
    if args.resume_path is None: # build a new dir if training from scratch
        embedding_save_path = args.data_path + '/embeddings_' + strftime("%m%d%H%M%S", localtime())
        os.mkdir(embedding_save_path)
    else: # resume training using original dir
        embedding_save_path = args.resume_path
    print("embedding_save_path: ", embedding_save_path)

    with open(embedding_save_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load data splits
    data_split = np.load(args.data_path + '/data_split.npy')
        
    # Loss
    if args.use_hardest_neg:
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))
    if args.use_dgi:
        loss_fn_dgi = nn.BCEWithLogitsLoss()

    # Metrics
    metrics=[AverageNonzeroTripletsMetric()] # Counts average number of nonzero triplets found in minibatches


    train_i = 0 # Initially, only use block 0 as training set (with explicit labels)

    # Train on initial graph
    # Resume model from the initial block or start the experiment from scratch. Otherwise (to resume from other blocks) skip this step.
    if ((args.resume_path is not None) and (args.resume_point == 0) and (args.resume_current)) or args.resume_path is None:
        if not args.use_dgi:
            train_indices, indices_to_remove, model = initial_maintain(train_i, 0, data_split, metrics, embedding_save_path, loss_fn)
        else:
            train_indices, indices_to_remove, model = initial_maintain(train_i, 0, data_split, metrics, embedding_save_path, loss_fn, None, loss_fn_dgi)

    # Initialize the model, train_indices and indices_to_remove to avoid errors
    if args.resume_path is not None:
        model = None
        train_indices = None
        indices_to_remove = []

    # iterate through all blocks
    for i in range(1, data_split.shape[0]):
        # Inference (prediction)
        # Resume model from the previous, i.e., (i-1)th block or continue the new experiment. Otherwise (to resume from other blocks) skip this step.
        if ((args.resume_path is not None) and (args.resume_point == i-1) and (not args.resume_current)) or args.resume_path is None:
            if not args.use_dgi:
                model = infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, train_indices, model, None, indices_to_remove)
            else:
                model = infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, train_indices, model, loss_fn_dgi, indices_to_remove)
        # Maintain
        # Resume model from the current, i.e., ith block or continue the new experiment. Otherwise (to resume from other blocks) skip this step.
        if ((args.resume_path is not None) and (args.resume_point == i) and (args.resume_current)) or args.resume_path is None:
            if i % args.window_size == 0:
                train_i = i
                if not args.use_dgi:
                    train_indices, indices_to_remove, model = initial_maintain(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model)
                else:
                    train_indices, indices_to_remove, model = initial_maintain(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model, loss_fn_dgi)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper parameters
    parser.add_argument('--n_epochs', default=15, type=int,
                        help="Number of initial-training/maintenance-training epochs.")
    parser.add_argument('--n_infer_epochs', default=0, type=int,
                        help="Number of inference epochs.")                    
    parser.add_argument('--window_size', default=3, type=int,
                        help="Maintain the model after predicting window_size blocks.")
    parser.add_argument('--patience', default=5, type=int,
                        help="Early stop if performance did not improve in the last patience epochs.")                   
    parser.add_argument('--margin', default=3., type=float,
                        help="Margin for computing triplet losses")          
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Learning rate")  
    parser.add_argument('--batch_size', default=2000, type=int,
                        help="Batch size (number of nodes sampled to compute triplet loss in each batch)") 
    parser.add_argument('--n_neighbors', default=800, type=int,
                        help="Number of neighbors sampled for each node.") 
    parser.add_argument('--hidden_dim', default=8, type=int,
                        help="Hidden dimension")
    parser.add_argument('--out_dim', default=32, type=int,
                        help="Output dimension of tweet representations")  
    parser.add_argument('--num_heads', default=4, type=int,
                        help="Number of heads in each GAT layer")          
    parser.add_argument('--use_residual', dest='use_residual', default=True,
                        action='store_false',
                        help="If true, add residual(skip) connections")
    parser.add_argument('--validation_percent', default=0.2, type=float,
                        help="Percentage of validation nodes(tweets)")
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False,
                        action='store_true',
                        help="If true, use hardest negative messages to form triplets. Otherwise use random ones")
    parser.add_argument('--use_dgi', dest='use_dgi', default=False,
                        action='store_true',
                        help="If true, add a DGI term to the loss. Otherwise use triplet loss only")
    parser.add_argument('--remove_obsolete', default=2, type=int,
                        help="If 0, keep inseting new message blocks and never remove;\n"+ 
                        "if 1, remove obsolete training nodes (that are connected to the new messages arrived during the last window) during maintenance;\n"+
                        "if 2, during each prediction, use only the new data to construct message graph, during each maintenance, use only the last message block arrived during the last window for continue training.")

    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=False,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--data_path', default= './incremental_0808/incremental_graphs_1007', 
                        type=str, help="Path of features, labels and edges")
                        # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--mask_path', default= None, 
                        type=str, help="File path that contains the training, validation and test masks")     
                        # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--resume_path', default= None, 
                        type=str, help="File path that contains the partially performed experiment that needs to be resume.")
    parser.add_argument('--resume_point', default= 0, type=int, 
                        help="The block model to be loaded.")         
    parser.add_argument('--resume_current', dest='resume_current', default=True,
                        action='store_false', 
                        help="If true, continue to train the resumed model of the current block(to resume a partally trained initial/mantenance block);\
                            If false, start the next(infer/predict) block from scratch;")                                            
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval") 

    args = parser.parse_args()

    main(args)

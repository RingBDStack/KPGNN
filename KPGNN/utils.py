import dgl
import numpy as np
import torch
from scipy import sparse
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from main import args_define

args = args_define.args
# Dataset
class SocialDataset(Dataset):
    def __init__(self, path, index):
        self.features = np.load(path + '/' + str(index) + '/features.npy')
        temp = np.load(path + '/' + str(index) + '/labels.npy', allow_pickle=True)
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
    def remove_obsolete_nodes(self, indices_to_remove=None):  # indices_to_remove: list
        # torch.range(0, (self.labels.shape[0] - 1), dtype=torch.long)
        if indices_to_remove is not None:
            all_indices = np.arange(0, self.labels.shape[0]).tolist()
            indices_to_keep = list(set(all_indices) - set(indices_to_remove))
            self.features = self.features[indices_to_keep, :]
            self.labels = self.labels[indices_to_keep]
            self.matrix = self.matrix[indices_to_keep, :]  # keep row
            self.matrix = self.matrix[:, indices_to_keep]  # keep column
            #  remove nodes from matrix


# Compute the representations of all the nodes in g using model
def extract_embeddings(g, model, num_all_samples, labels):
    with torch.no_grad():
        model.eval()
        for batch_id, nf in enumerate(
                dgl.contrib.sampling.NeighborSampler(g,  # sample from the whole graph (contain unseen nodes)
                                                     num_all_samples,  # set batch size = the total number of nodes
                                                     1000,
                                                     # set the expand_factor (the number of neighbors sampled from
                                                     # the neighbor list of a vertex) to None: get error: non-int
                                                     # expand_factor not supported
                                                     neighbor_type='in',
                                                     shuffle=False,
                                                     num_workers=32,
                                                     num_hops=2)):
            nf.copy_from_parent()
            if args.use_dgi:
                extract_features, _ = model(nf)  # representations of all nodes
            else:
                extract_features = model(nf)  # representations of all nodes
            extract_nids = nf.layer_parent_nid(-1).to(device=extract_features.device, dtype=torch.long)  # node ids
            extract_labels = labels[extract_nids]  # labels of all nodes
        assert batch_id == 0
        extract_nids = extract_nids.data.cpu().numpy()
        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()
        # generate train/test mask
        A = np.arange(num_all_samples)
        # print("A", A)
        assert (A == extract_nids).all()

    return extract_nids, extract_features, extract_labels


def save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, path, counter):
    np.savetxt(path + '/features_' + str(counter) + '.tsv', extract_features, delimiter='\t')
    np.savetxt(path + '/labels_' + str(counter) + '.tsv', extract_labels, fmt='%i', delimiter='\t')
    with open(path + '/labels_tags_' + str(counter) + '.tsv', 'w') as f:
        f.write('label\tmessage_id\ttrain_tag\n')
        for (label, mid, train_tag) in zip(extract_labels, extract_nids, extract_train_tags):
            f.write("%s\t%s\t%s\n" % (label, mid, train_tag))
    print("Embeddings after inference epoch " + str(counter) + " saved.")
    print()


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def run_kmeans(extract_features, extract_labels, indices, isoPath=None):
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

    # k-means clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)

    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, nmi)


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


def graph_statistics(G, save_path):
    message = '\nGraph statistics:\n'

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges / 2) // num_nodes
    in_degrees = G.in_degrees()
    isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = (in_degrees == isolated_nodes)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt')
    num_isolated_nodes = torch.sum(isolated_nodes).item()

    message += 'We have ' + str(num_nodes) + ' nodes.\n'
    message += 'We have ' + str(num_edges / 2) + ' in-edges.\n'
    message += 'Average degree: ' + str(ave_degree) + '\n'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "a") as f:
        f.write(message)

    return num_isolated_nodes



def generateMasks(length, data_split, train_i, i, validation_percent=0.2, save_path=None, num_indices_to_remove=0):
    """
        Intro:
        This function generates train and validation indices for initial/maintenance epochs and test indices for inference(prediction) epochs
        If remove_obsolete mode 0 or 1:
        For initial/maintenance epochs:
        - The first (train_i + 1) blocks (blocks 0, ..., train_i) are used as training set (with explicit labels)
        - Randomly sample validation_percent of the training indices as validation indices
        For inference(prediction) epochs:
        - The (i + 1)th block (block i) is used as test set
        Note that other blocks (block train_i + 1, ..., i - 1) are also in the graph (without explicit labels, only their features and structural info are leveraged)
        If remove_obsolete mode 2:
        For initial/maintenance epochs:
        - The (i + 1) = (train_i + 1)th block (block train_i = i) is used as training set (with explicit labels)
        - Randomly sample validation_percent of the training indices as validation indices
        For inference(prediction) epochs:
        - The (i + 1)th block (block i) is used as test set

        :param length: the length of label list
        :param data_split: loaded splited data (generated in custom_message_graph.py)
        :param train_i, i: flag, indicating for initial/maintenance stage if train_i == i and inference stage for others
        :param validation_percent: the percent of validation data occupied in whole dataset
        :param save_path: path to save data
        :param num_indices_to_remove: number of indices ought to be removed

        :returns train indices, validation indices or test indices
    """
    if args.remove_obsolete == 0 or args.remove_obsolete == 1:  # remove_obsolete mode 0 or 1
        # verify total number of nodes
        assert length == (np.sum(data_split[:i + 1]) - num_indices_to_remove)

        # If is in initial/maintenance epochs, generate train and validation indices
        if train_i == i:
            # randomly shuffle the training indices
            train_length = np.sum(data_split[:train_i + 1])
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
        # If the process is in inference(prediction) epochs, generate test indices
        else:
            test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
            test_indices += (np.sum(data_split[:i]) - num_indices_to_remove)
            if save_path is not None:
                torch.save(test_indices, save_path + '/test_indices.pt')
                test_indices = torch.load(save_path + '/test_indices.pt')
            return test_indices

    else:  # remove_obsolete mode 2
        # verify total number of nodes
        assert length == data_split[i]

        # If is in initial/maintenance epochs, generate train and validation indices
        if train_i == i:
            # randomly shuffle the graph indices
            train_indices = torch.randperm(length)
            # get total number of validation indices
            n_validation_samples = int(length * validation_percent)
            # sample n_validation_samples validation indices and use the rest as training indices
            validation_indices = train_indices[:n_validation_samples]
            train_indices = train_indices[n_validation_samples:]
            if save_path is not None:
                torch.save(validation_indices, save_path +
                           '/validation_indices.pt')
                torch.save(train_indices, save_path + '/train_indices.pt')
                validation_indices = torch.load(
                    save_path + '/validation_indices.pt')
                train_indices = torch.load(save_path + '/train_indices.pt')
            return train_indices, validation_indices
        # If is in inference(prediction) epochs, generate test indices
        else:
            test_indices = torch.range(
                0, (data_split[i] - 1), dtype=torch.long)
            if save_path is not None:
                torch.save(test_indices, save_path + '/test_indices.pt')
                test_indices = torch.load(save_path + '/test_indices.pt')
            return test_indices


# Utility function, finds the indices of the values' elements in tensor
def find(tensor, values):
    return torch.nonzero(tensor.cpu()[..., None] == values.cpu())

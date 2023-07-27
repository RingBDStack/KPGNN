import torch
import dgl
import numpy as np
import torch
from scipy import sparse
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from dgl.data import DGLDataset


def run_kmeans(extract_features, extract_labels, indices=None, isoPath=None):
    if not indices is None:
        # Extract the features and labels of the test tweets
        indices = indices.cpu().detach().numpy()

        # if isoPath is not None:
        #     # Remove isolated points
        #     temp = torch.load(isoPath)
        #     temp = temp.cpu().detach().numpy()
        #     non_isolated_index = list(np.where(temp != 1)[0])
        #     indices = intersection(indices, non_isolated_index)

        # Extract labels
        labels_true = extract_labels[indices]
        # Extract features
        X = extract_features[indices, :].cpu().detach().numpy()
    else:
        labels_true = extract_labels
        X = extract_features.cpu().detach().numpy()
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    # n_classes = len(set(list(labels_true)))
    n_classes = len(labels_true.unique())
    # print(n_classes)

    # k-means clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    # centers = kmeans.cluster_centers_
    if isinstance(labels_true, torch.Tensor):
        labels_true = labels_true.cpu().detach().numpy()
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)

    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, nmi)

def run_kmeans_in_train(extract_features, extract_labels, indices=None, isoPath=None):
    if not indices is None:
        # Extract the features and labels of the test tweets
        indices = indices.cpu().detach().numpy()
        # if isoPath is not None:
        #     # Remove isolated points
        #     temp = torch.load(isoPath)
        #     temp = temp.cpu().detach().numpy()
        #     non_isolated_index = list(np.where(temp != 1)[0])
        #     indices = intersection(indices, non_isolated_index)

        # Extract labels
        labels_true = extract_labels[indices]
        # Extract features
        X = extract_features[indices, :]
    else:
        labels_true = extract_labels
        X = extract_features.cpu().detach().numpy()
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    # n_classes = len(set(list(labels_true)))
    n_classes = len(labels_true.unique())
    # print(n_classes)

    # k-means clustering
    kmeans = KMeans(n_clusters=n_classes, n_init="auto", random_state=0).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    labels_true = labels_true.cpu()
    # labels_to_onehot = torch.nn.functional.one_hot(torch.tensor(labels).to(torch.int64))
    # labels_to_onehot = labels_to_onehot.to(torch.float)
    # centers_of_features = torch.mm(labels_to_onehot, torch.tensor(centers))
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, centers, nmi)

def run_dbscan(extract_features, extract_labels, indices, isoPath=None):
    indices = indices.cpu().detach().numpy()

    # if isoPath is not None:
    #     # Remove isolated points
    #     temp = torch.load(isoPath)
    #     temp = temp.cpu().detach().numpy()
    #     non_isolated_index = list(np.where(temp != 1)[0])
    #     indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]
    n_classes = len(labels_true.unique())

    hdbscan = HDBSCAN().fit(X)
    labels = hdbscan.labels_
    print(labels)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)

    predicted_classes = len(set(labels)) - (1 if -1 in labels else 0)
    print("Predicted Classes: ", predicted_classes)

    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, nmi)


def run_hdbscan(extract_features, extract_labels, indices, isoPath=None):
    indices = indices.cpu().detach().numpy()

    # if isoPath is not None:
    #     # Remove isolated points
    #     temp = torch.load(isoPath)
    #     temp = temp.cpu().detach().numpy()
    #     non_isolated_index = list(np.where(temp != 1)[0])
    #     indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]
    n_classes = len(labels_true.unique())

    # k-means clustering
    dbscan = DBSCAN(eps=.4).fit(X)
    labels = dbscan.labels_
    print(labels)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)

    predicted_classes = len(set(labels)) - (1 if -1 in labels else 0)
    print("Predicted Classes: ", predicted_classes)

    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, nmi)

def run_dbscan_in_train(extract_features, extract_labels, indices=None, isoPath=None):
    if not indices is None:
        # Extract the features and labels of the test tweets
        indices = indices.cpu().detach().numpy()
        # if isoPath is not None:
        #     # Remove isolated points
        #     temp = torch.load(isoPath)
        #     temp = temp.cpu().detach().numpy()
        #     non_isolated_index = list(np.where(temp != 1)[0])
        #     indices = intersection(indices, non_isolated_index)

        # Extract labels
        labels_true = extract_labels[indices]
        # Extract features
        X = extract_features[indices, :]
    else:
        labels_true = extract_labels
        X = extract_features.cpu().detach().numpy()
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    # n_classes = len(set(list(labels_true)))
    n_classes = len(labels_true.unique())
    # print(n_classes)

    # k-means clustering
    dbscan = DBSCAN(eps=.1).fit(X)
    labels = dbscan.labels_
    centers = dbscan.components_
    print(dbscan.labels_)
    labels_true = labels_true.cpu()
    # labels_to_onehot = torch.nn.functional.one_hot(torch.tensor(labels).to(torch.int64))
    # labels_to_onehot = labels_to_onehot.to(torch.float)
    # centers_of_features = torch.mm(labels_to_onehot, torch.tensor(centers))
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, centers, nmi)

def run_hdbscan_in_train(extract_features, extract_labels, indices=None, isoPath=None):
    if not indices is None:
        # Extract the features and labels of the test tweets
        indices = indices.cpu().detach().numpy()
        # if isoPath is not None:
        #     # Remove isolated points
        #     temp = torch.load(isoPath)
        #     temp = temp.cpu().detach().numpy()
        #     non_isolated_index = list(np.where(temp != 1)[0])
        #     indices = intersection(indices, non_isolated_index)

        # Extract labels
        labels_true = extract_labels[indices]
        # Extract features
        X = extract_features[indices, :]
    else:
        labels_true = extract_labels
        X = extract_features.cpu().detach().numpy()
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    # n_classes = len(set(list(labels_true)))
    n_classes = len(labels_true.unique())
    # print(n_classes)

    # k-means clustering
    hdbscan = HDBSCAN(min_cluster_size=2, store_centers="centroid").fit(X)
    labels = hdbscan.labels_
    centers = hdbscan.centroids_
    labels_true = labels_true.cpu()
    # labels_to_onehot = torch.nn.functional.one_hot(torch.tensor(labels).to(torch.int64))
    # labels_to_onehot = labels_to_onehot.to(torch.float)
    # centers_of_features = torch.mm(labels_to_onehot, torch.tensor(centers))
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, centers, nmi)

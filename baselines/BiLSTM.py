import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# The pack_padded_sequence is a format that enables the model to ignore the padded elements.
# The pad_packed_sequence function is a reversed operation for pack_padded_sequence and 
# will bring the output back to the familiar format [batch_size, sentence_length, hidden_features]
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import os
import pandas as pd
from collections import Counter
from itertools import combinations
from time import time
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import en_core_web_lg
from sklearn.cluster import KMeans
from sklearn import metrics
import random

lr = 1e-3
batch_size = 1000
dropout_keep_prob = 0.8
embedding_size = 300
max_size = 5000 # maximum vocabulary size
seed = 1
num_hidden_nodes = 32
hidden_dim2 = 64
num_layers = 1  # LSTM layers
bi_directional = True #False 
pad_index = 0
num_epochs = 20
margin = 3 # Margin for computing triplet losses
max_len = 10 # truncate all the tweets after max_len th word


class LSTM(nn.Module):

    # define all the layers used in model
    def __init__(self, embedding_dim, weight, lstm_units, hidden_dim , lstm_layers,
                 bidirectional, dropout, pad_index, batch_size):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        # use pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(weight, padding_idx = pad_index)
        self.lstm = nn.LSTM(embedding_dim,
                            lstm_units,
                            num_layers=lstm_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        num_directions = 2 if bidirectional else 1
        self.fc1 = nn.Linear(lstm_units * num_directions, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm_layers = lstm_layers
        self.num_directions = num_directions
        self.lstm_units = lstm_units


    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units)),
                Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units)))
        return h, c

    def forward(self, text, text_lengths):
        batch_size = text.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        # output of shape (batch, seq_len, num_directions * hidden_size): tensor containing the 
        # output features (h_t) from the last layer of the LSTM, for each t.
        output, (h_n, c_n) = self.lstm(packed_embedded, (h_0, c_0)) 
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        # get the hidden state of the last time step 
        out = output_unpacked[:, -1, :]
        rel = self.relu(out)
        dense1 = self.fc1(rel)
        #drop = self.dropout(dense1)
        #preds = self.fc2(drop)
        preds = self.dropout(dense1)
        return preds

"""
# unpadded dataset
class VectorizeData(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.df.wordsidx[idx]
        y = self.df.event_id[idx]
        return x,y
"""

# padded dataset
class VectorizeData(Dataset):
    def __init__(self, df, maxlen = max_len):
        self.df = df
        self.maxlen = maxlen
        print("Calculating lengths")
        # truncate the tweets
        self.df["lengths"] = self.df.wordsidx.apply(lambda x: self.maxlen if len(x) > self.maxlen else len(x))
        print('Filter out rows with lengths equal to 0')
        print('Original shape: ', self.df.shape)
        self.df = self.df[self.df["lengths"] > 0].reset_index(drop=True)
        print('Shape after filtering: ', self.df.shape)
        print("Padding")
        self.df["wordsidxpadded"] = self.df.wordsidx.apply(self.pad_data)
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.df.wordsidxpadded[idx]
        lens = self.df.lengths[idx] # truncated tweet length
        y = self.df.event_type_ids[idx]
        sample = {'text':(x,lens), 'label':y}
        return sample
    
    def pad_data(self, tweet):
        padded = np.zeros((self.maxlen,), dtype = np.int64)
        if len(tweet) > self.maxlen:
            padded[:] = tweet[:self.maxlen]
        else:
            padded[:len(tweet)] = tweet
        return padded


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

# print average loss every log_interval batches
def train(model, train_iterator, optimizer, loss_func, log_interval = 40):
    n_batches = len(train_iterator)
    epoch_loss = 0
    for i, batch in enumerate(train_iterator):
        optimizer.zero_grad()
        text, text_lengths = batch['text'] 
        predictions = model(text, text_lengths)
        loss, num_triplets = loss_func(predictions, batch['label'])#.squeeze())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if i % log_interval == 0:
            print(f'\tBatch: [{i}/{n_batches} ({100. * (i+1) / n_batches:.0f}%)]\tLoss: {epoch_loss / (i+1):.4f}\tNum_triplets: {num_triplets}')
    return epoch_loss / n_batches

def run_kmeans(features, labels):
    assert features.shape[0] == labels.shape[0]
    # Get the total number of classes
    n_classes = len(set(labels.tolist()))
    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(features)
    pred_labels = kmeans.labels_
    nmi = metrics.normalized_mutual_info_score(labels, pred_labels)
    ami = metrics.adjusted_mutual_info_score(labels, pred_labels)
    ari = metrics.adjusted_rand_score(labels, pred_labels)
    return nmi, ami, ari

def evaluate(model, test_iterator):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            assert i == 0 # cluster all the test tweets at once
            text, text_lengths = batch['text']
            predictions = model(text, text_lengths)
            validate_nmi, validate_ami, validate_ari = run_kmeans(predictions, batch['label'])#.squeeze())
    return validate_nmi, validate_ami, validate_ari

def run_train(epochs, model, train_iterator, test_iterator, optimizer, loss_func):
    all_nmi, all_ami, all_ari = [], [], []
    for epoch in range(epochs):
        # train the model
        start = time()
        print(f'Epoch {epoch}. Training.')
        train_loss = train(model, train_iterator, optimizer, loss_func)
        print(f'\tTrain Loss: {train_loss:.4f}')
        print(f'\tThis epoch took {(time() - start)/60:.2f} mins to train.')
        # evaluate the model
        start = time()
        print(f'Epoch {epoch}. Evaluating.')
        validate_nmi, validate_ami, validate_ari = evaluate(model, test_iterator) ######
        all_nmi.append(validate_nmi)
        all_ami.append(validate_ami)
        all_ari.append(validate_ari)
        print(f'\tVal. NMI: {validate_nmi:.4f}')
        print(f'\tVal. AMI: {validate_ami:.4f}')
        print(f'\tVal. ARI: {validate_ari:.4f}')
        print(f'\tThis epoch took {(time() - start)/60:.2f} mins to evaluate.')
    return all_nmi, all_ami, all_ari

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    p = './'
    s_p = './'

    # load MAVEN dataset
    df_np = np.load(s_p + 'all_df_words_ents_mids.npy', allow_pickle=True)
    print("Data loaded.")
    df = pd.DataFrame(data=df_np, \
        columns=['document_ids', 'sentence_ids', 'sentences', 'event_type_ids', 'words', 'unique_words', 'entities', 'message_ids'])
    print("df_np converted to dataframe.")

    # load tokenized tweets
    f_batch_text = df.iloc[:, 5]
    print("Extracted tweets.")

    # count unique words (converted to lowercases)
    words = Counter()
    for tweet in f_batch_text.values:
        #print(tweet)
        words.update(w.lower() for w in tweet)
    #print(len(words))
    # convert words from counter to list (sorted by frequencies from high to low)
    words = [key for key, _ in words.most_common()]
    # Add _PAD and _UNK token at the begining
    words = ['_PAD','_UNK'] + words
    #print(len(words))
    print('Extracted unique words.')

    # construct a mapping of words to indicies and vice versa
    word2idx = {o:i for i,o in enumerate(words)}
    idx2word = {i:o for i,o in enumerate(words)}
    # save
    np.save(p + 'word2idx.npy', word2idx) 
    np.save(p + 'idx2word.npy', idx2word) 
    print('Constructed and saved word2idx and idx2word maps.')
    
    # Load
    word2idx = np.load(p + 'word2idx.npy',allow_pickle='TRUE').item()
    #idx2word = np.load(p + 'idx2word.npy',allow_pickle='TRUE').item()
    #print(word2idx)
    #print(idx2word)
    print('word2idx map loaded.')

    # convert tokenized tweets to indicies
    def indexer(tweet): return [word2idx[w.lower()] for w in tweet]
    df["wordsidx"] = df.words.apply(indexer)
    print('Tokenized tweets in the df to word indices.')

    # load pretrained word embeddings
    weight = np.zeros((len(word2idx), embedding_size), dtype = np.float)

    # Load pre-trained word2vec model
    start = time()
    nlp = en_core_web_lg.load()
    print('Word2vec model took ', (time() - start)/60, ' mins to load.')

    # update word embeddings to weight
    for i in range(len(word2idx)):
        w = idx2word.get(i)
        w = nlp(w)
        #print(w.vector)
        if w.has_vector:
            weight[i] = w.vector
    print('Word embeddings extracted. Shape: ', weight.shape)

    # save and load word embeddings
    np.save(p + 'word_embeddings.npy', weight)
    print('Word embeddings saved.')
    weight = np.load(p + 'word_embeddings.npy')
    print('Word embeddings loaded. Shape: ', weight.shape)
    weight = torch.tensor(weight, dtype=torch.float)

    # split training and test datasets
    # load masks
    train_mask = list(np.load(s_p + 'train_indices_7170.npy'))
    test_mask = list(np.load(s_p + 'test_indices_2048.npy'))
    #print(len(train_mask))
    #print(len(test_mask))
    train_data = VectorizeData(df.iloc[train_mask, :].copy().reset_index(drop=True))
    test_data = VectorizeData(df.iloc[test_mask, :].copy().reset_index(drop=True))
    
    # construct training and test iterator
    #train_iterator = create_iterator(train_data, batch_size, device)
    train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle = True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    test_iterator = DataLoader(test_data, batch_size=len(test_data), shuffle = True)

    # loss function
    loss_func = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))

    # model
    lstm_model = LSTM(embedding_size, weight, num_hidden_nodes, hidden_dim2, num_layers, bi_directional, dropout_keep_prob, pad_index, batch_size)

    # optimizer
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

    # train and evaluation
    all_nmi, all_ami, all_ari = run_train(num_epochs, lstm_model, train_iterator, test_iterator, optimizer, loss_func)
    best_epoch = [i for i, j in enumerate(all_nmi) if j == max(all_nmi)][0]
    print("all_nmi: ", all_nmi)
    print("all_ami: ", all_ami)
    print("all_ari: ", all_ari)
    print("\nExperiment completed. Best results at epoch ", best_epoch)
    print(f"num_epochs = {num_epochs}, max_len = {max_len}, NMI = {all_nmi[best_epoch]:.4f}, AMI = {all_ami[best_epoch]:.4f}, ARI = {all_ari[best_epoch]:.4f}")
    
    
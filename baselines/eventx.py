# eventx original paper:
# Bang Liu, Fred X Han, Di Niu, Linglong Kong, Kunfeng Lai, and Yu Xu. 2020. Story Forest: Extracting Events and Telling Stories from Breaking News. TKDD 14, 3 (2020), 1–28.
import pandas as pd
import numpy as np
import itertools
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import random
from sklearn import metrics
from collections import Counter
from statistics import mean
import os
import json
import pickle
import time
import torch


def construct_offline_df(load_path, mask_path, save_path):
    #load MAVEN dataset
    all_df_words_ents_mids_np = np.load(load_path, allow_pickle=True)
    all_df_words_ents_mids = pd.DataFrame(data=all_df_words_ents_mids_np, \
        columns=['document_ids', 'sentence_ids', 'sentences', 'event_type_ids', 'words', 'unique_words', 'entities', 'message_ids'])
    print("Loaded all_df_words_ents_mid.")
    print("Data converted to dataframe.")
    
    # load test indices
    test_mask = np.load('./test_indices_2048.npy')

    df = all_df_words_ents_mids.iloc[test_mask, :]
    print("Test df extracted.")
    #print(df.head(10))
    #print()

    np.save(save_path + '/corpus_offline.npy', df.values)
    print("corpus_offline saved.")


def construct_dict(df, dir_path = None):
    kw_pair_dict = {} # contains all pairs(sorted) of keywords/entities and their corresponding enclosing tweets' ids as values
    kw_dict = {} # contains all distinct keywords and entities as keys, and their corresponding enclosing tweets' ids as values

    for _, row in df.iterrows():
        tweet_id = str(row['message_ids'])

        entities = row['entities']
        entities = ['_'.join(tup) for tup in entities]
        for each in entities:
            if each not in kw_dict.keys():
                kw_dict[each] = []
            kw_dict[each].append(tweet_id)

        words = row['unique_words']
        for each in words:
            if each not in kw_dict.keys():
                kw_dict[each] = []
            kw_dict[each].append(tweet_id)
        
        #for r in itertools.product(entities, words):
        for r in itertools.combinations(entities + words, 2):
            r = list(r)
            r.sort()
            pair = (r[0], r[1])
            if pair not in kw_pair_dict.keys():
                kw_pair_dict[pair] = []
            kw_pair_dict[pair].append(tweet_id)

    if dir_path is not None:
        pickle.dump(kw_dict, open(dir_path + '/kw_dict.pickle','wb'))
        pickle.dump(kw_pair_dict, open(dir_path + '/kw_pair_dict.pickle','wb'))

    return kw_pair_dict, kw_dict

# index the keywords/entities (for more convinient downstream tf-idf embedding)
def map_dicts(kw_pair_dict, kw_dict, dir_path = None):
    map_index_to_kw = {}
    m_kw_dict = {}
    for i, k in enumerate(kw_dict.keys()):
        map_index_to_kw['k'+str(i)] = k
        m_kw_dict['k'+str(i)] = kw_dict[k]
    map_kw_to_index = {v:k for k,v in map_index_to_kw.items()}
    m_kw_pair_dict = {}
    for _, pair in enumerate(kw_pair_dict.keys()):
        m_kw_pair_dict[(map_kw_to_index[pair[0]], map_kw_to_index[pair[1]])] = kw_pair_dict[pair]

    if dir_path is not None:
        pickle.dump(m_kw_pair_dict, open(dir_path + '/m_kw_pair_dict.pickle','wb'))
        pickle.dump(m_kw_dict, open(dir_path + '/m_kw_dict.pickle','wb'))
        pickle.dump(map_index_to_kw, open(dir_path + '/map_index_to_kw.pickle','wb'))
        pickle.dump(map_kw_to_index, open(dir_path + '/map_kw_to_index.pickle','wb'))

    return m_kw_pair_dict, m_kw_dict, map_index_to_kw, map_kw_to_index

def construct_kw_graph(kw_pair_dict, kw_dict, min_cooccur_time, min_prob):
    G = nx.Graph()
    # add nodes
    G.add_nodes_from(list(kw_dict.keys()))
    # add edges between pairs of keywords that can meet the 2 conditions
    for pair, co_tid_list in kw_pair_dict.items():
        if (len(co_tid_list) > min_cooccur_time):
            #print('condition 1 met')
            #print(pair, co_tid_list)
            if (len(co_tid_list)/len(kw_dict[pair[0]]) > min_prob) and (len(co_tid_list)/len(kw_dict[pair[1]]) > min_prob):
                #print('condition 2 met')
                #print(pair[0], kw_dict[pair[0]])
                #print(pair[1], kw_dict[pair[1]])
                G.add_edge(*pair)
            #print()

    return G

# recursive version, can cause RecursionError when running on large graphs. Changed to iterative version below.
def detect_kw_communities(G, communities, kw_pair_dict, kw_dict, max_kw_num = 3):
    connected_components = [ c for c in nx.connected_components(G)]
    if len(connected_components) >= 1:
        c = connected_components[0]
        if len(c) < max_kw_num:
            communities.append(c)
            G.remove_nodes_from(c)
        else:
            c_sub_G = G.subgraph(c).copy()
            d = nx.edge_betweenness_centrality(c_sub_G)
            max_value = max(d.values())
            edges = [key for key, value in d.items() if value == max_value]
            # If two edges have the same betweenness score, the one with lower conditional probability will be removed
            if len(edges) > 1:
                probs = []
                for e in edges:
                    e = list(e)
                    e.sort()
                    pair = (e[0], e[1])
                    co_len = len(kw_pair_dict[pair])
                    e_prob = (co_len/len(kw_dict[pair[0]]) + co_len/len(kw_dict[pair[1]]))/2
                    probs.append(e_prob)
                min_prob = min(probs)
                min_index = [i for i, j in enumerate(probs) if j == min_prob]
                edge_to_remove = edges[min_index[0]]
            else:
                edge_to_remove = edges[0]
            G.remove_edge(*edge_to_remove)
        detect_kw_communities(G, communities, kw_pair_dict, kw_dict, max_kw_num)
    else:
        return

# iterative version
def detect_kw_communities_iter(G, communities, kw_pair_dict, kw_dict, max_kw_num = 3):
    connected_components = [ c for c in nx.connected_components(G)]
    while len(connected_components) >= 1:
        c = connected_components[0]
        if len(c) < max_kw_num:
            communities.append(c)
            G.remove_nodes_from(c)
        else:
            c_sub_G = G.subgraph(c).copy()
            d = nx.edge_betweenness_centrality(c_sub_G)
            max_value = max(d.values())
            edges = [key for key, value in d.items() if value == max_value]
            # If two edges have the same betweenness score, the one with lower conditional probability will be removed
            if len(edges) > 1:
                probs = []
                for e in edges:
                    e = list(e)
                    e.sort()
                    pair = (e[0], e[1])
                    co_len = len(kw_pair_dict[pair])
                    e_prob = (co_len/len(kw_dict[pair[0]]) + co_len/len(kw_dict[pair[1]]))/2
                    probs.append(e_prob)
                min_prob = min(probs)
                min_index = [i for i, j in enumerate(probs) if j == min_prob]
                edge_to_remove = edges[min_index[0]]
            else:
                edge_to_remove = edges[0]
            G.remove_edge(*edge_to_remove)
        connected_components = [ c for c in nx.connected_components(G)]

def map_communities(communities, map_kw_to_index):
    m_communities = []
    for cluster in communities:
        m_cluster = ' '.join(map_kw_to_index[kw] for kw in cluster)
        m_communities.append(m_cluster)
    return m_communities


def classify_docs(test_tweets, m_communities, map_kw_to_index, dir_path = None):

    m_test_tweets = []
    for doc in test_tweets:
        #print(doc)
        m_doc = ' '.join(map_kw_to_index[kw] for kw in doc)
        m_test_tweets.append(m_doc)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(m_communities + m_test_tweets)
    #print(m_test_tweets)
    #a1 = cosine_similarity(m_test_tweets[0], X[0])
    #print(a1)
    train_size = len(m_communities)
    test_size = len(m_test_tweets)
    classes = []
    for i in range(test_size):
        #print(i)
        cosine_similarities = linear_kernel(X[train_size + i], X[:train_size]).flatten()
        max_similarity = cosine_similarities[cosine_similarities.argsort()[-1]]
        related_clusters = [i for i, sim in enumerate(cosine_similarities) if sim == max_similarity]
        if len(related_clusters) == 1:
            classes.append(related_clusters[0])
        else:
            classes.append(random.choice(related_clusters))
    
    if dir_path is not None:
        np.save(dir_path + '/classes.npy', classes)
        
    return classes

def map_tweets(df, dir_path = None):
    m_tweets = []
    ground_truths = []
    for _, row in df.iterrows():
        entities = row['entities']
        entities = ['_'.join(tup) for tup in entities]
        words = row['unique_words']
        m_tweets.append(entities + words)
        ground_truths.append(row['event_type_ids'])
    
    if dir_path is not None:
        np.save(dir_path + '/m_tweets.npy', m_tweets) # a list of lists of keywords (entities and sampled_words)
        np.save(dir_path + '/ground_truths.npy', ground_truths)

    return m_tweets, ground_truths

def check_class_sizes(ground_truths, predictions):
    #distinct_true_labels = list(Counter(ground_truths).keys()) # equals to list(set(ground_truths))
    count_true_labels = list(Counter(ground_truths).values()) # counts the elements' frequency
    ave_true_size = mean(count_true_labels)
    
    distinct_predictions = list(Counter(predictions).keys()) # equals to list(set(ground_truths))
    count_predictions = list(Counter(predictions).values()) # counts the elements' frequency

    large_classes = [distinct_predictions[i] for i,count in enumerate(count_predictions) if count > ave_true_size]

    return large_classes

def main(): 

    s_p = './'
    root_save_path = './'
    num_repeats = 5
    
    # use only the test data defined by mask
    # construct and save offline test df
    print("Start constructing test df ...") 
    construct_offline_df(s_p + 'all_df_words_ents_mids.npy', s_p + 'test_indices_2048.npy', root_save_path)

    # load offline test df
    df_np = np.load(root_save_path + '/corpus_offline.npy', allow_pickle=True)
    print("Data offline loaded.")
    df = pd.DataFrame(data=df_np, \
        columns=['document_ids', 'sentence_ids', 'sentences', 'event_type_ids', 'words', 'unique_words', 'entities', 'message_ids'])
    print("Data converted to dataframe.")

    # repeat the experiment for num_repeats times
    all_ars, all_ami, all_nmi = [], [], [] # record the ars, ami, and nmi of all experiments
    all_time = [] # record time spent on predicting each experiment

    for i in range(num_repeats):
        print("========================================")
        start = time.time()
        save_path = root_save_path + '/' + str(i)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    
        kw_pair_dict, kw_dict = construct_dict(df, save_path)
        m_kw_pair_dict, m_kw_dict, map_index_to_kw, map_kw_to_index = map_dicts(kw_pair_dict, kw_dict, save_path)

        min_cooccur_time = 2 # condition one: the times of co-occurrence shall be above a minimum threshold min_cooccur_time
        min_prob = 0.15 # condition two: the conditional probabilities of the occurrence Pr{wj|wi} and Pr{wi|wj} also need to be greater than a predefined threshold min_prob

        # construct keyword graph. Use all keywords as nodes and add edges between pairs that met the above two conditions
        G = construct_kw_graph(kw_pair_dict, kw_dict, min_cooccur_time, min_prob)
        #connected_c_lengths = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        #print(len(connected_c_lengths))
        #print()

        max_kw_num = 3 # the splitting process ends if the number of nodes in each subgraph is smaller than a predefined threshold max_kw_num
        communities = []
        # split the keyword graph into clusters (stored in communities, a list of lists of nodes that belong to the same cluster)
        #detect_kw_communities(G, communities, kw_pair_dict, kw_dict, max_kw_num = max_kw_num)
        detect_kw_communities_iter(G, communities, kw_pair_dict, kw_dict, max_kw_num = max_kw_num)
        # save the keyword clusters
        np.save(save_path + '/communities.npy', communities)
        #print(communities)
        #community_lengths = [len(c) for c in communities]
        #print(community_lengths)

        # map the nodes in each keyword cluster to an encoded format for easier tf-idf representing latter
        m_communities = map_communities(communities, map_kw_to_index)
        # save the mapped keyword clusters
        np.save(save_path + '/m_communities.npy', m_communities)
        #print(m_communities)
        m_tweets, ground_truths = map_tweets(df, save_path)
        #print(m_tweets)
        classes = classify_docs(m_tweets, m_communities, map_kw_to_index, save_path)
        #print(classes)
        ars = metrics.adjusted_rand_score(ground_truths, classes)
        print("Adjusted random score: ", ars)
        ami = metrics.adjusted_mutual_info_score(ground_truths, classes)
        print("Adjusted Mutual Information: ", ami)
        nmi = metrics.normalized_mutual_info_score(ground_truths, classes)
        print("Normalized Mutual Information: %0.3f"%nmi)
        # save ars, ami, and nmi
        all_ars.append(ars)
        all_ami.append(ami)
        all_nmi.append(nmi)
        np.save(save_path + '/all_ars.npy', all_ars)
        np.save(save_path + '/all_ami.npy', all_ami)
        np.save(save_path + '/all_nmi.npy', all_nmi)

        # record time
        end = time.time()
        t = (end-start)/60
        print('Eventx took ', t, ' mins')
        all_time.append(t)
        np.save(save_path + '/all_time.npy', all_time)
        print()

    print("Done predicting all the experiments.")


if __name__=="__main__": 
    main() 
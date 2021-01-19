
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
from scipy import sparse
import dgl
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
import pickle
from collections import Counter
from time import time
import os

# construct a graph using tweet ids, user ids, entities and rare (sampled) words (4 modalities)
# if G is not None then insert new nodes to G
def construct_graph_from_df(df, G = None):
    if G is None:
        G = nx.Graph()
    for _, row in df.iterrows():
        tid = 't_' + str(row['tweet_id'])
        G.add_node(tid)
        G.node[tid]['tweet_id'] = True # right-hand side value is irrelevant for the lookup

        user_ids = row['user_mentions']
        user_ids.append(row['user_id'])
        user_ids = ['u_' + str(each) for each in user_ids]
        #print(user_ids)
        G.add_nodes_from(user_ids)
        for each in user_ids:
            G.node[each]['user_id'] = True

        entities = row['entities']
        #entities = ['e_' + each for each in entities]
        #print(entities)
        G.add_nodes_from(entities)
        for each in entities:
            G.node[each]['entity'] = True

        words = row['sampled_words']
        words = ['w_' + each for each in words]
        #print(words)
        G.add_nodes_from(words)
        for each in words:
            G.node[each]['word'] = True

        edges = []
        edges += [(tid, each) for each in user_ids]
        edges += [(tid, each) for each in entities]
        edges += [(tid, each) for each in words]
        G.add_edges_from(edges)

    return G

# convert networkx graph to dgl graph and store its sparse binary adjacency matrix
def to_dgl_graph_v3(G, save_path = None):

    message = ''
    print('Start converting heterogeneous networkx graph to homogeneous dgl graph.')
    message += 'Start converting heterogeneous networkx graph to homogeneous dgl graph.\n'
    all_start = time()

    print('\tGetting a list of all nodes ...')
    message += '\tGetting a list of all nodes ...\n'
    start = time()
    all_nodes = list(G.nodes)
    mins = (time() - start)/60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    #print('All nodes: ', all_nodes)
    #print('Total number of nodes: ', len(all_nodes))

    print('\tGetting adjacency matrix ...')
    message += '\tGetting adjacency matrix ...\n'
    start = time()
    A = nx.to_numpy_matrix(G) # Returns the graph adjacency matrix as a NumPy matrix.
    mins = (time() - start)/60 
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # compute commuting matrices
    print('\tGetting lists of nodes of various types ...')
    message += '\tGetting lists of nodes of various types ...\n'
    start = time()
    tid_nodes = list(nx.get_node_attributes(G, 'tweet_id').keys())
    userid_nodes = list(nx.get_node_attributes(G, 'user_id').keys())
    word_nodes = list(nx.get_node_attributes(G, 'word').keys())
    entity_nodes = list(nx.get_node_attributes(G, 'entity').keys())
    del G
    mins = (time() - start)/60 
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\tConverting node lists to index lists ...')
    message += '\tConverting node lists to index lists ...\n'
    start = time()
    indices_tid = [all_nodes.index(x) for x in tid_nodes]
    indices_userid = [all_nodes.index(x) for x in userid_nodes]
    indices_word = [all_nodes.index(x) for x in word_nodes]
    indices_entity = [all_nodes.index(x) for x in entity_nodes]
    del tid_nodes
    del userid_nodes
    del word_nodes
    del entity_nodes
    mins = (time() - start)/60 
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # tweet-user-tweet
    print('\tStart constructing tweet-user-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-user matrix ...')
    message += '\tStart constructing tweet-user-tweet commuting matrix ...\n\t\t\tStart constructing tweet-user matrix ...\n'
    start = time()
    w_tid_userid = A[np.ix_(indices_tid, indices_userid)]
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_userid = sparse.csr_matrix(w_tid_userid)
    del w_tid_userid
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_userid_tid = s_w_tid_userid.transpose()
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-user * user-tweet ...')
    message += '\t\t\tCalculating tweet-user * user-tweet ...\n'
    start = time()
    s_m_tid_userid_tid = s_w_tid_userid * s_w_userid_tid
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_userid_tid.npz", s_m_tid_userid_tid)
        print("Sparse binary userid commuting matrix saved.")
        del s_m_tid_userid_tid
    del s_w_tid_userid
    del s_w_userid_tid
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'
    
    # tweet-ent-tweet
    print('\tStart constructing tweet-ent-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-ent matrix ...')
    message += '\tStart constructing tweet-ent-tweet commuting matrix ...\n\t\t\tStart constructing tweet-ent matrix ...\n'
    start = time()
    w_tid_entity = A[np.ix_(indices_tid, indices_entity)]
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_entity = sparse.csr_matrix(w_tid_entity)
    del w_tid_entity
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_entity_tid = s_w_tid_entity.transpose()
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-ent * ent-tweet ...')
    message += '\t\t\tCalculating tweet-ent * ent-tweet ...\n'
    start = time()
    s_m_tid_entity_tid = s_w_tid_entity * s_w_entity_tid
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_entity_tid.npz", s_m_tid_entity_tid)
        print("Sparse binary entity commuting matrix saved.")
        del s_m_tid_entity_tid
    del s_w_tid_entity
    del s_w_entity_tid
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # tweet-word-tweet
    print('\tStart constructing tweet-word-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-word matrix ...')
    message += '\tStart constructing tweet-word-tweet commuting matrix ...\n\t\t\tStart constructing tweet-word matrix ...\n'
    start = time()
    w_tid_word = A[np.ix_(indices_tid, indices_word)]
    del A
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_word = sparse.csr_matrix(w_tid_word)
    del w_tid_word
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_word_tid = s_w_tid_word.transpose()
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-word * word-tweet ...')
    message += '\t\t\tCalculating tweet-word * word-tweet ...\n'
    start = time()
    s_m_tid_word_tid = s_w_tid_word * s_w_word_tid
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_word_tid.npz", s_m_tid_word_tid)
        print("Sparse binary word commuting matrix saved.")
        del s_m_tid_word_tid
    del s_w_tid_word
    del s_w_word_tid
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # compute tweet-tweet adjacency matrix
    print('\tComputing tweet-tweet adjacency matrix ...')
    message += '\tComputing tweet-tweet adjacency matrix ...\n'
    start = time()
    if save_path is not None:
        s_m_tid_userid_tid = sparse.load_npz(save_path + "s_m_tid_userid_tid.npz")
        print("Sparse binary userid commuting matrix loaded.")
        s_m_tid_entity_tid = sparse.load_npz(save_path + "s_m_tid_entity_tid.npz")
        print("Sparse binary entity commuting matrix loaded.")
        s_m_tid_word_tid = sparse.load_npz(save_path + "s_m_tid_word_tid.npz")
        print("Sparse binary word commuting matrix loaded.")
    
    s_A_tid_tid = s_m_tid_userid_tid + s_m_tid_entity_tid
    del s_m_tid_userid_tid
    del s_m_tid_entity_tid
    s_bool_A_tid_tid = (s_A_tid_tid + s_m_tid_word_tid).astype('bool')
    del s_m_tid_word_tid
    del s_A_tid_tid
    mins = (time() - start)/60 
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'
    all_mins = (time() - all_start)/60
    print('\tOver all time elapsed: ', all_mins, ' mins\n')
    message += '\tOver all time elapsed: '
    message += str(all_mins)
    message += ' mins\n'
    
    if save_path is not None:
        sparse.save_npz(save_path + "s_bool_A_tid_tid.npz", s_bool_A_tid_tid)
        print("Sparse binary adjacency matrix saved.")
        s_bool_A_tid_tid = sparse.load_npz(save_path + "s_bool_A_tid_tid.npz")
        print("Sparse binary adjacency matrix loaded.")
    
    # create corresponding dgl graph
    G = dgl.DGLGraph(s_bool_A_tid_tid)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    print()
    message += 'We have '
    message += str(G.number_of_nodes())
    message += ' nodes.'
    message += 'We have '
    message += str(G.number_of_edges())
    message += ' edges.\n'

    return all_mins, message

def construct_incremental_dataset_0922(df, save_path, features, test = True):
    # If test equals true, construct the initial graph using test_ini_size tweets
    # and increment the graph by test_incr_size tweets each day
    test_ini_size = 500
    test_incr_size = 100

    # save data splits for training/validate/test mask generation
    data_split = []
    # save time spent for the heterogeneous -> homogeneous conversion of each graph
    all_graph_mins = []
    message = ""
    # extract distinct dates
    distinct_dates = df.date.unique()
    #print("Distinct dates: ", distinct_dates)
    print("Number of distinct dates: ", len(distinct_dates))
    print()
    message += "Number of distinct dates: "
    message += str(len(distinct_dates))
    message += "\n"

    # split data by dates and construct graphs
    # first week -> initial graph (20254 tweets)
    print("Start constructing initial graph ...")
    message += "\nStart constructing initial graph ...\n"
    ini_df = df.loc[df['date'].isin(distinct_dates[:7])]
    if test:
        ini_df = ini_df[:test_ini_size]
    G = construct_graph_from_df(ini_df)
    path = save_path + '0/'
    os.mkdir(path)
    grap_mins, graph_message = to_dgl_graph_v3(G, save_path = path)
    message += graph_message
    print("Initial graph saved")
    message += "Initial graph saved\n"
    # record the total number of tweets
    data_split.append(ini_df.shape[0])
    # record the time spent for graph conversion
    all_graph_mins.append(grap_mins)
    # extract and save the labels of corresponding tweets
    y = ini_df['event_id'].values
    y = [int(each) for each in y]
    np.save(path + 'labels.npy', np.asarray(y))
    print("Labels saved.")
    message += "Labels saved.\n"
    # extract and save the features of corresponding tweets
    indices = ini_df['index'].values.tolist()
    x = features[indices, :]
    np.save(path + 'features.npy', x)
    print("Features saved.")
    message += "Features saved.\n\n"

    # subsequent days -> insert tweets day by day (skip the last day because it only contains one tweet)
    for i in range(7, len(distinct_dates)-1):
        print("Start constructing graph ", str(i-6), " ...")
        message += "\nStart constructing graph "
        message += str(i-6)
        message += " ...\n"
        incr_df = df.loc[df['date'] == distinct_dates[i]]
        if test:
            incr_df = incr_df[:test_incr_size]
        #G = construct_graph_from_df(incr_df, G) 
        G = construct_graph_from_df(incr_df) # remove obsolete, version 2: construct graph using only the data of the day
        path = save_path + str(i-6) + '/'
        os.mkdir(path)
        grap_mins, graph_message = to_dgl_graph_v3(G, save_path = path)
        message += graph_message
        print("Graph ", str(i-6), " saved")
        message += "Graph "
        message += str(i-6)
        message += " saved\n"
        # record the total number of tweets
        data_split.append(incr_df.shape[0])
        # record the time spent for graph conversion
        all_graph_mins.append(grap_mins)
        # extract and save the labels of corresponding tweets
        #y = np.concatenate([y, incr_df['event_id'].values], axis = 0)
        y = [int(each) for each in incr_df['event_id'].values]
        np.save(path + 'labels.npy', y)
        print("Labels saved.")
        message += "Labels saved.\n"
        # extract and save the features of corresponding tweets
        indices = incr_df['index'].values.tolist()
        x = features[indices, :]
        #x = np.concatenate([x, x_incr], axis = 0)
        np.save(path + 'features.npy', x)
        print("Features saved.")
        message += "Features saved.\n"
    
    return message, data_split, all_graph_mins

save_path = './incremental_graphs_0922_test/'

# load data (68841 tweets, multiclasses filtered)
df_np = np.load('./68841_tweets_multiclasses_filtered_0722.npy', allow_pickle=True)
print("Data loaded.")
df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
    "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
    "words", "filtered_words", "sampled_words"])
print("Data converted to dataframe.")
#print(df.shape)
#print(df.head(10))
#print()

# sort data by time
df = df.sort_values(by='created_at').reset_index()

# append date
df['date'] = [d.date() for d in df['created_at']]

#print(df['date'].value_counts())

# load features
f = np.load('../features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy')

# generate test graphs, features, and labels
message, data_split, all_graph_mins = construct_incremental_dataset_0922(df, save_path, f, True)
with open(save_path + "node_edge_statistics.txt", "w") as text_file:
    text_file.write(message)
np.save(save_path + 'data_split.npy', np.asarray(data_split))
print("Data split: ", data_split)
np.save(save_path + 'all_graph_mins.npy', np.asarray(all_graph_mins))
print("Time sepnt on heterogeneous -> homogeneous graph conversions: ", all_graph_mins)

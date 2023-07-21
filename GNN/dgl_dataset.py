from dgl.data import DGLBuiltinDataset
import numpy as np
import pandas as pd
import networkx as nx
import en_core_web_lg
from datetime import datetime
from time import time
import dgl
import os
import torch

class TwitterDataset(DGLBuiltinDataset):

    def __init__(self, savepath=None, split=0, raw_dir=None, force_reload=False, verbose=True):
        self.raw_data = "/data/lby/social-event-detection/KPGNN/datasets/Twitter/"
        self.split = split
        self.predict_category = "tweet"
        self.savepath = os.path.join(savepath, str(split))
        super(TwitterDataset, self).__init__(   name="twitter",
                                                url=None,
                                                raw_dir=raw_dir,
                                                force_reload=force_reload,
                                                verbose=verbose)
    def load_df(self):
        p_part1 = self.raw_data + '68841_tweets_multiclasses_filtered_0722_part1.npy'
        p_part2 = self.raw_data + '68841_tweets_multiclasses_filtered_0722_part2.npy'
        df_np_part1 = np.load(p_part1, allow_pickle=True)
        df_np_part2 = np.load(p_part2, allow_pickle=True)
        df_np = np.concatenate((df_np_part1, df_np_part2), axis = 0)
        print("Loaded data.")
        df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
            "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
            "words", "filtered_words", "sampled_words"])
        # sort data by time
        df = df.sort_values(by='created_at').reset_index()

        # append date
        df['date'] = [d.date() for d in df['created_at']]
        print("Data converted to dataframe.")
        print(df.shape)
        print(df.head(10))
        return df
    
    def extract_time_feature(self, t_str):
        t = datetime.fromisoformat(str(t_str))
        OLE_TIME_ZERO = datetime(1899, 12, 30)
        delta = t - OLE_TIME_ZERO
        return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

    # encode the times-tamps of all the messages in the dataframe
    def df_to_t_features(self, df):
        t_features = np.asarray([self.extract_time_feature(t_str) for t_str in df['created_at']])
        return t_features

    def construct_graph_from_df(self, df, savepath=None, G=None):
        start = time()
        if os.path.exists(os.path.join(savepath,"nx_graph.gpickle")):
            print("Loading existed nx graph")
            return nx.read_gpickle(os.path.join(savepath,"nx_graph.gpickle"))
        if G is None:
            G = nx.Graph()
        nlp = en_core_web_lg.load()
        self.num_classes = len(set(df["event_id"]))
        for ind, row in df.iterrows():
            tid = 't_' + str(row['tweet_id'])
            label = row['event_id']
            t_features = np.asarray(self.extract_time_feature(row['created_at']))
            G.add_node(tid)
            G.nodes[tid]['type'] = 'tweet'  # right-hand side value is irrelevant for the lookup
            G.nodes[tid]['origin_index'] = ind
            G.nodes[tid]['features'] = np.concatenate((nlp(' '.join(row['filtered_words'])).vector, t_features))
            G.nodes[tid]['label'] = label

            # The last one is the user posting the tweet
            user_ids = row['user_mentions']
            user_ids.append(row['user_id'])
            user_ids = ['u_' + str(each) for each in user_ids]
            # print(user_ids)
            G.add_nodes_from(user_ids)
            for each in user_ids:
                G.nodes[each]['type'] = 'user'
                G.nodes[each]['origin_index'] = ind
                G.nodes[each]['features'] = int(each[3:])

            hashtags = row['hashtags']
            hashtags = list(set(['h_' + each for each in hashtags]))
            G.add_nodes_from(hashtags)
            for shift, each in enumerate(hashtags):
                G.nodes[each]['type'] = 'hashtag'
                G.nodes[each]['origin_index'] = ind
                G.nodes[each]['shift'] = shift
                G.nodes[each]['features'] = nlp(row['hashtags'][shift]).vector

            words = row['sampled_words']
            words = ['w_' + each.lower() for each in words]
            # print(words)
            G.add_nodes_from(words)
            for shift, each in enumerate(words):
                G.nodes[each]['type'] = 'word'
                G.nodes[each]['origin_index'] = ind
                G.nodes[each]['shift'] = shift
                G.nodes[each]['features'] = nlp(row['words'][shift]).vector
            
            # loc = 'l_' + row['user_loc']
            # G.add_node(loc)
            # G.node[loc]['location']=True

            uid = user_ids[-1]
            edges = []
            edges += [(tid, each) for each in user_ids]
            edges += [(uid, each) for each in user_ids[:-1]]
            edges += [(tid, each) for each in hashtags]
            edges += [(tid, each) for each in words]
            # edges += [(uid, loc)]
            G.add_edges_from(edges)
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')

        start = time()
        print("Writing nx graph to " + os.path.join(savepath,"nx_graph.gpickle"))
        nx.write_gpickle(G, os.path.join(savepath,"nx_graph.gpickle"))
        mins = (time() - start) / 60
        print('Done. Time elapsed: ', mins, ' mins\n')
        print('Constructed nx graph with {} nodes, {} edges, {} tweets and {} types of events.'.format(\
            len(G.nodes), len(G.edges), len(df), self.num_classes))
        return G
    
    def nx_to_dgl(self, G):
        message = ''
        print('Start converting heterogeneous networkx graph to homogeneous dgl graph.')
        message += 'Start converting heterogeneous networkx graph to homogeneous dgl graph.\n'
        all_start = time()

        print('\tGetting a list of all nodes ...')
        message += '\tGetting a list of all nodes ...\n'
        start = time()
        all_nodes = list(G.nodes)
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'
        print('\tGetting a list of all edges ...')
        message += '\tGetting a list of all edges ...\n'
        start = time()
        all_edges = list(G.edges)
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'
        start = time()
        print('\tGetting nodes of all_types')

        tweet_nodes = [each for each in all_nodes if G.nodes[each]["type"]=="tweet"]
        user_nodes = [each for each in all_nodes if G.nodes[each]["type"]=="user"]
        hashtag_nodes = [each for each in all_nodes if G.nodes[each]["type"]=="hashtag"]
        word_nodes = [each for each in all_nodes if G.nodes[each]["type"]=="word"]

        tweet_features = np.array([G.nodes[each]["features"] for each in tweet_nodes])
        user_features = np.array([G.nodes[each]["features"] for each in user_nodes])
        hashtag_features = np.array([G.nodes[each]["features"] for each in hashtag_nodes])
        word_features = np.array([G.nodes[each]["features"] for each in word_nodes])
        features = {"tweet": tweet_features, "user": user_features, "hashtag": hashtag_features, "word": word_features}
        labels = np.array([G.nodes[each]["label"] for each in tweet_nodes])
        
        nodes_type = {"tweet": tweet_nodes, "user": user_nodes, "hashtag": hashtag_nodes, "word": word_nodes}
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)

        dic = {}
        print('\tInterating edges to generate dictionary for heterograph ...')
        message += '\tInterating edges to generate dictionary for heterograph ...\n'
        start = time()
        relation_lookup = {("tweet", "user"): "t-u", ("user", "user"): "u-u", ("tweet", "hashtag"): "t-h", ("tweet", "word"): "t-w",\
                           ("user", "tweet"): "u-t", ("hashtag", "tweet"): "h-t", ("word", "tweet"): "w-t"}
        for edge in all_edges:
            hnode, tnode = edge
            htype = G.nodes[hnode]["type"]
            ttype = G.nodes[tnode]["type"]
            hindex = nodes_type[htype].index(hnode)
            tindex = nodes_type[ttype].index(tnode)
            relation = relation_lookup[(htype, ttype)]
            if (htype, relation, ttype) in dic:
                dic[(htype, relation, ttype)][0].append(hindex)
                dic[(htype, relation, ttype)][1].append(tindex)
            else:
                dic[(htype, relation, ttype)] = ([hindex], [tindex])
        # 组织成{(类型，关系，类型)：（[节点1,节点2,...]，[节点1, 节点2, ...]）}
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'
        dgl_graph = dgl.heterograph(dic)
        start = time()
        print('\tAssign node labels and masks.')
        for type in dgl_graph.ntypes:
            dgl_graph.nodes[type].data['features'] = torch.tensor(features[type])
        dgl_graph.nodes["tweet"].data['labels'] = torch.tensor(labels)
        dgl_graph.nodes['tweet'].data['train_mask'] = torch.ones(len(tweet_nodes), dtype=torch.bool)
        # dgl_graph.nodes['tweet'].data['test_mask'] = ~dgl_graph.nodes['tweet'].data['train_mask']
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')

        mins = (time() - all_start) / 60
        print('Done DGL graph construction. Time elapsed: ', mins, ' mins\n')
        
        return dgl_graph

    def process(self):
        message = ""
        df = self.load_df()

        # 划分数据集
        distinct_dates = df.date.unique()  # 2012-11-07
        # print("Distinct dates: ", distinct_dates)
        print("Number of distinct dates: ", len(distinct_dates))
        print()
        message += "Number of distinct dates: "
        message += str(len(distinct_dates))
        message += "\n"

        # split data by dates and construct graphs
        # first week -> initial graph (20254 tweets)
        print("Start constructing initial graph ...")
        message += "\nStart constructing initial graph ...\n"
        if self.split == 0:
            df = df.loc[df['date'].isin(distinct_dates[:7])]  # find top 7 dates
        else:
            df = df.loc[df['date'] == distinct_dates[self.split+7]]
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath, exist_ok=False)
        G = self.construct_graph_from_df(df, savepath=self.savepath)
        # 构建图
        self._g = self.nx_to_dgl(G)
        # self._g = dgl.reorder_graph(g)

    def __getitem__(self, idx):
        assert idx == 0, "这个数据集里只有一个图"
        return self._g

    def __len__(self):
        return 1
    def save(self):
        graph_path = os.path.join(self.savepath, 'dgl_graph.bin')
        dgl.save_graphs(graph_path, [self._g])
    def load(self):
        # 从目录 `self.save_path` 里读取处理过的数据
        graph_path = os.path.join(self.savepath, 'dgl_graph.bin')
        graphs, _ = dgl.load_graphs(graph_path)
        self._g = graphs[0]
        self.num_classes = 506

    def has_cache(self):
        # 检查在 `self.save_path` 里是否有处理过的数据文件
        graph_path = os.path.join(self.savepath, 'dgl_graph.bin')
        return os.path.exists(graph_path)
    
if __name__ == "__main__":
    for i in range(0, 22):
        dataset = TwitterDataset("./incremental_graph/", split=i)
        print(dataset._g.number_of_nodes(), dataset._g.number_of_edges())
    # dataset.process()
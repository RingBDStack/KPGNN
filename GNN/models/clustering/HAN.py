import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl.sampling import RandomWalkNeighborSampler


class RelGraphEmbed_Pretrain(nn.Module):
    r"""Embedding layer for featureless heterograph."""

    def __init__(
        self, g, embed_size, embed_name="embed", activation=None, dropout=0.0
    ):
        super(RelGraphEmbed_Pretrain, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.embeds = {}
        for ntype in g.ntypes:
            embed = None
            if ntype != "user":
                embed = nn.Linear(g.nodes[ntype].data["features"].shape[1], embed_size)
            else:
                embed = nn.Embedding(g.num_nodes(ntype), self.embed_size)
            # nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
            self.embeds[ntype] = embed

    def forward(self, block=None):
        """Forward computation

        Parameters
        ----------
        block : DGLGraph, optional
            If not specified, directly return the full graph with embeddings stored in
            :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
            graph and return.

        Returns
        -------
        DGLGraph
            The block graph fed with embeddings.
        """
        h = {}
        if block is None:
            for ntype in self.embeds:
                if ntype != "user":
                    h[ntype] = self.embeds[ntype](self.g.nodes[ntype].data["features"])
                else:
                    h[ntype] = self.embeds[ntype](self.g.nodes(ntype))
        else:
            for ntype in self.embeds:
                if ntype != "user":
                    h[ntype] = self.embeds[ntype](block.nodes[ntype].data["features"])
                else:
                    h[ntype] = self.embeds[ntype](block.nodes(ntype))

        return h

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)
class HANLayer(torch.nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_metapath : number of metapath based sub-graph
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
        self, num_metapath, in_size, out_size, layer_num_heads, dropout
    ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapath):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_metapath = num_metapath

    def forward(self, block_list, h_list):
        semantic_embeddings = []
        for i, block in enumerate(block_list):
            semantic_embeddings.append(
                self.gat_layers[i](block, h_list[i]).flatten(1)
            )
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(
        self, g, num_metapath, embed_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        # self.embed_layer=RelGraphEmbed_Pretrain(g, embed_size)
        self.layers.append(
            HANLayer(num_metapath, embed_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    num_metapath,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        # h = self.embed_layer(g)
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)
    
class HANSampler(object):
    def __init__(self, g, metapath_list, num_neighbors):
        self.sampler_list = []
        for metapath in metapath_list:
            # note: random walk may get same route(same edge), which will be removed in the sampled graph.
            # So the sampled graph's edges may be less than num_random_walks(num_neighbors).
            self.sampler_list.append(
                RandomWalkNeighborSampler(
                    G=g,
                    num_traversals=1,
                    termination_prob=0,
                    num_random_walks=num_neighbors,
                    num_neighbors=num_neighbors,
                    metapath=metapath,
                )
            )

    def sample_blocks(self, seeds):
        block_list = []
        for sampler in self.sampler_list:
            frontier = sampler(seeds)
            # add self loop
            frontier = dgl.remove_self_loop(frontier)
            frontier.add_edges(torch.tensor(seeds), torch.tensor(seeds))
            block = dgl.to_block(frontier, seeds)
            block_list.append(block)
        return seeds, block_list


# KPGNN

This repository contains the source code and preprocessed dataset for The Web Conference 2021 paper [Knowledge-Preserving Incremental Social Event Detection via Heterogeneous GNNs](https://arxiv.org/pdf/2101.08747.pdf).

# To run KPGNN

Step 1) cd /KPGNN

Step 2) run *generate_initial_features.py* to generate the initial features for the messages (please see Figure 1(b) and Section 3.2 of the paper for more details).

Step 3) run *custom_message_graph.py* to construct incremental message graphs. To construct small message graphs for test purpose, set *test=True* when calling *construct_incremental_dataset_0922()*. To use all the messages (see Table. 4 of the paper for a statistic of the number of messages in the graphs), set *test=False*.

Step 4) run *main.py*

# Datasets

## Twitter dataset
The Twitter dataset [1] is collected to evaluate social event detection methods. 
After filtering out repeated and irretrievable tweets, the dataset contains 68,841 manually labeled tweets 
related to 503 event classes, spread over a period of four weeks. 
Please find the original dataset at http://mir.dcs.gla.ac.uk/resources/

## MAVEN dataset
MAVEN [2] is a general domain event detection dataset constructed from Wikipedia documents. 
We remove sentences (i.e., messages) that are associated with multiple event types. 
The filtered dataset contains 10,242 messages related to 154 event classes.
Please find the original dataset at https://github.com/THU-KEG/MAVEN-dataset

## Data format and usage
Please refer to [this document](https://github.com/RingBDStack/KPGNN/blob/main/datasets/data_usage.md) for data format and usage.

# Baselines
For Word2vec[3], we use the [spaCy pre-trained vectors](https://spacy.io/models/en#en_core_web_lg).

For [LDA](https://radimrehurek.com/gensim/models/ldamodel.html)[4], [WMD](https://tedboy.github.io/nlps/generated/generated/gensim.similarities.WmdSimilarity.html#gensim.similarities.WmdSimilarity)[5], [BERT](https://github.com/huggingface/transformers)[6], and [PP-GCN](https://github.com/RingBDStack/PPGCN)[7], we use the open-source implementations.

We implement EventX[8] with Python 3.7.3 and BiLSTM[9] with Pytorch 1.6.0. Please refer to the baselines folder. 

# Citation
If you find this repository helpful, please consider citing the following paper.

```bibtex
@article{cao2021knowledge,
  title={Knowledge-Preserving Incremental Social Event Detection via Heterogeneous GNNs},
  author={Cao, Yuwei and Peng, Hao and Wu, Jia and Dou, Yingtong and Li, Jianxin and Yu, Philip S},
  journal={arXiv preprint arXiv:2101.08747},
  year={2021}
}
```




[1] Andrew J McMinn, Yashar Moshfeghi, and Joemon M Jose. 2013. Building a large-scale corpus for evaluating event detection on twitter. In Proceedings of the CIKM.ACM, 409–418.

[2] Xiaozhi Wang, Ziqi Wang, Xu Han, Wangyi Jiang, Rong Han, Zhiyuan Liu, Juanzi Li, Peng Li, Yankai Lin, and Jie Zhou. 2020. MAVEN: A Massive General Domain
Event Detection Dataset. In Proceedings of EMNLP.

[3] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. In Proceedings of ICLR.

[4] David M Blei, Andrew Y Ng, and Michael I Jordan. 2003. Latent dirichlet allocation. JMLR 3, Jan (2003), 993–1022.

[5] Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. 2015. From word embeddings to document distances. In Proceedings of the ICML. 957–966.

[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).

[7] Hao Peng, Jianxin Li, Qiran Gong, Yangqiu Song, Yuanxing Ning, Kunfeng Lai, and Philip S. Yu. 2019. Fine-grained event categorization with heterogeneous graph convolutional networks. In Proceedings of the IJCAI. 3238–3245.

[8] Bang Liu, Fred X Han, Di Niu, Linglong Kong, Kunfeng Lai, and Yu Xu. 2020. Story Forest: Extracting Events and Telling Stories from Breaking News. TKDD 14, 3 (2020), 1–28.

[9] Alex Graves and Jürgen Schmidhuber. 2005. Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural networks 18, 5-6 (2005), 602–610.

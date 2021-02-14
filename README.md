
# KPGNN

This repository contains the source code and preprocessed dataset for The Web Conference 2021 paper [Knowledge-Preserving Incremental Social Event Detection via Heterogeneous GNNs](https://arxiv.org/pdf/2101.08747.pdf).

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
Please refer to this document for data format and usage.

# Baselines
Coming soon.

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

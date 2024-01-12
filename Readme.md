# Inferring gene regulatory networks based on graph embedding Inferring gene regulatory networks inference from single-cell transcriptomics based on graph embedding

## Abstract

Gene regulatory networks (GRNs) encode gene regulation in living organisms, and have become a critical tool to understand complex biological processes. However, due to the dynamic and complex nature of gene regulation, inferring GRNs from scRNA-seq data is still a challenging task. Existing computational methods usually focus on the close connections between genes, and ignore the global structure and distal regulatory relationships. Here, we develop a supervised deep learning framework, IGEGRNS, to infer gene regulatory networks from scRNA-seq data based on graph embedding. In the framework, contextual information of genes is captured by GraphSAGE, which aggregates gene features and neighborhood structures to generate low-dimensional embedding for genes. Then, the k most influential nodes in the whole graph are filtered through top-k pooling. Finally, potential regulatory relationships between genes are predicted by stacking CNNs. Compared with eight competing supervised and unsupervised methods, our method achieves better performance on six time-series scRNA-seq datasets.
![image-20240112194233539](https://github.com/DHUDBlab/IGEGRNS/assets/93750046/e6778871-1c16-4c34-b7a2-ad9da4b21922)
![image-20240112194304209](https://github.com/DHUDBlab/IGEGRNS/assets/93750046/623e73ad-c218-4472-84f7-6fac3b6f9cd9)


## Install Application:

1.Anaconda (https://www.anaconda.com/) 2.Git(https://github.com/)

## Enviorment:

pythorch==1.8.0

pyg==2.0.1

pandas==0.25.3

numpy==1.19.5

## Running

python Main.py

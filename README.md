# Hybrid CUDA Solution for AllPairs Similarity Search on High-dimensional Sparse Datasets 
Given a set of high dimensional sparse vectors, a similarity function and a threshold, AllPairs Similarity Search finds out all pairs of vectors whose similarity values are higher than or equal to the threshold. AllPairs Similarity Search (APSS) has been studied in many different fields of computer science, including information retrieval, data mining, database and so on. It is a crucial part of lots of applications, such as near-duplicate document detection, collaborative filtering, query refinement and clustering.

For cosine similarity, many serial algorithms have been proposed to solve the problem by decreasing the possible similarity candidates for each query object. However, the efficiency of those serial algorithms degrade badly as the threshold decreases. Other parallel implementations of APSS based on OpenMP or MapReduce also adopt the pruning policy and do not solve the problem thoroughly. 

In this context, we introduce CuAPSS, which solves the All Pairs cosine similarity search problem in CUDA environment on GPUs. Our method adopts a hybrid method to utilize both forward list and inverted list in APSS which compromises between the memory visiting and dot-product computing. The experimental results show that our method could solve the problem much faster than existing methods on several benchmark datasets with hundreds of millions of non-zero values, achieving the speedup of 1.5X--23X against the state-of-the-art parallel algorithm, while keep a relatively stable running time with different values of the threshold.

Our contribution are as follows,

- Proposed a parallel cosine similarity search algorithm on Word2Vec using CUDA, combining a feature-parallel scan and a pair-parallel scan on different parts of vectors, trading off between memory accessing and dot product computing.
- Proposed a parameter tuning method for a user-defined p parameter for best performance based on statistical characteristics of high-dimensional sparse datasets, avoiding many atomic operations.
- Implemented on GeForce GTX 1080, achieving 14X-85X speedup over cuSPARSE and 1.5X−23X speedup over the SOTA multi-threading implementation, maintaining a relatively stable running time with different similarity threshold.


Please site our publication when using **APSS-CUDA** in your work.
```
@InProceedings{10.1007/978-3-030-05051-1_29,
author="Feng, Yilin and Tang, Jie and Wang, Chongjun and Xie, Junyuan",
title="CuAPSS: A Hybrid CUDA Solution for AllPairs Similarity Search",
booktitle="Algorithms and Architectures for Parallel Processing",
year="2018",
publisher="Springer International Publishing",
address="Cham",
pages="421--436"}
```

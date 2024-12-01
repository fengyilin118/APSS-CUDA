# Hybrid CUDA Solution for AllPairs Similarity Search on High-dimensional Sparse Datasets 
- Proposed a parallel cosine similarity search algorithm on Word2Vec using CUDA, combining a feature-parallel scan and a pair-parallel scan on different parts of vectors, trading off between memory accessing and dot product computing.
-	Proposed a parameter tuning method for a user-defined p parameter for best performance based on statistical characteristics of high-dimensional sparse datasets, avoiding many atomic operations.
-	Implemented on GeForce GTX 1080, achieving 14X-85X speedup over cuSPARSE and 1.5X−23X speedup over the SOTA multi-threading implementation, maintaining a relatively stable running time with different similarity threshold.


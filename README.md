# ADA-NNS (NSG) - Angular Distance-Guided Neighbor Selection for Graph-Based Approximate Nearest Neighbor Search

This repository is HNSW with ADA-NNS. ADA-NNS is a novel guided search algorithm that dynamically filters out less relevant neighbors by employing a lightweight proxy based on an approximate angular distance

### Prerequisites

+ GCC 9.4.0+ with OpenMP
+ CMake 3.22.2+
+ Boost 1.55+
+ [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html)
+ Eigen library

**IMPORTANT NOTE: this code uses AVX-256 intructions for fast distance computation, so your machine MUST support AVX-256 intructions, this can be checked using `cat /proc/cpuinfo | grep avx2`.** 

### Datasets

| Name     | Dimension | No. of base | No. of query | Metric |
|----------|-----------|-------------|--------------|--------|
| [SIFT1M](http://corpus-texmex.irisa.fr/)   | 128       | 1,000,000   | 10,000       | L2 |
| [GIST1M](http://corpus-texmex.irisa.fr/)   | 960       | 1,000,000   | 1,000        | L2 |
| [CRAWL](http://github.com/ZJULearning/SSG)    | 300       | 1,989,995   | 10,000       | L2 |
| DEEP100M* | 96        | 100,000,000 | 10,000        | L2 |
+ For DEEP100M, we will share the file link upon request

### Dataset Conversion

For datasets provided in HDF5 format (e.g., GLOVE-100 and NYTIMES),  
Parse HDF5 and generate fvecs and ivecs as follows:  
```bash
python ./utils/hdf5_to_vecs.py [hdf5_file_name]
mkdir -p dataset/[dataset_name]
mv [dataset_name]_*vecs dataset/[dataset_name]
```

### Compile On Ubuntu

Install Dependencies:

```shell
sudo apt-get install g++ cmake libgoogle-perftools-dev libeigen3-dev
```

Compile NSG:
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
```

### Million-scale Tests Reproduction
Download datasets in the `dataset` directory

Build kNN Graph:  
You can use either [efanna\_graph](https://github.com/ZJULearning/efanna\_graph) or [faiss](https://github.com/facebookresearch/faiss) to build this kNN graph.

The parameters used [faiss](https://github.com/facebookresearch/faiss) to build each kNN graph is as follows:

| Dataset | K |
|---------|---|
| SIFT1M  | 200 |
| GIST1M  | 400 |
| CRAWL   | 400 |
| DEEP100M | 400 |

Build NSG index from kNN Graph:  
You can use following command to build NSG index:

```bash
cd [NSG_HOME]/build/tests
./test_nsg_index [dataset_path] [kNN_graph_path] [L] [R] [C] [nsg_index_path]
```
+ `L` controls the quality of the NSG, the larger the better.
+ `R` controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.
+ `C` controls the maximum candidate pool size during NSG contruction.

| Dataset | L | R | C |
|---------|---|---|---|
| SIFT1M  | 40 | 50 | 500 |
| GIST1M  | 60 | 70 | 500 |
| CRAWL   | 150 | 50 | 1000 |
| DEEP100M | 200 | 40 | 1000 |

These are parameters used to build NSG index.

To reproduce ADA-NNS (NSG) results:
```bash
cd [NSG_HOME]/build/tests
./test_nsg_optimized_search [dataset_path] [query_path] [groundtruth_path] [nsg_index_path] [search_L] [search_K] [result_path] [num_threads] [tau] [hash_bitwidth] 
```
+ `SEARCH_L` controls the quality of the search results, the larger the better but slower. The `SEARCH_L` cannot be samller than the `SEARCH_K`
+ `SEARCH_K` controls the number of result neighbors we want to query.

Following parameters are used to reproduce ADA-NNS results:

| Dataset | tau | hash_bitwidth |
|---------|-----|---------------|
| SIFT1M  | 0.2 | 512           |
| GIST1M  | 0.2 | 1024          |
| CRAWL   | 0.2 | 512           |
| DEEP100M | 0.2 | 512          |


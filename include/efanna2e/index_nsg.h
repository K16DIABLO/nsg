#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>

#include <Eigen/Dense>

namespace efanna2e {

class IndexNSG : public Index {
 public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexNSG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;
  void SearchWithOptGraph(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices,
      uint32_t* hashed_query_buffer);
  void OptimizeGraph(float* data);

  // Sungjun Jung: ADA-NNS functions
  void SetTau(const float tau) { tau_ = tau; }
  void SetHashBitwidth(const uint64_t hash_bitwidth) { hash_bitwidth_ = hash_bitwidth; }
  void GenerateHashFunction (char* file_name);
  void GenerateHashedSet (char* file_name, float* data);
  bool ReadHashFunction (char* file_name);
  bool ReadHashedSet (char* file_name);
  void QueryHash(const float* query, uint32_t* hashed_query, const uint64_t num_query);
  uint32_t CandidateSelection(const float query_norm, const uint32_t* hashed_query, std::vector<SimpleNeighbor>& selected_pool, boost::dynamic_bitset<>& flags, const uint32_t* neighbors);

  protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;

    Index *initializer_ = nullptr;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void tree_grow(const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);


  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_ = nullptr;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;

    // Sungjun Jung: Variables for ADA-NNS
    float tau_; // candidate selection threshold
                // _tau = 0.3 means only top 30% of neighbors 
                // having the smallest angular distance to the query are selected
    uint32_t hash_bitwidth_;
    float* hash_function_;
    uint32_t* hashed_set_;
};
}

#endif //EFANNA2E_INDEX_NSG_H

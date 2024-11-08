#include <Hornet.hpp>
#include "StandardAPI.hpp"
#include "Util/BatchFunctions.hpp"
#include "Util/RandomGraphData.cuh"
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <Device/Util/CudaUtil.cuh>     //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include <cuda_profiler_api.h>
#include <Core/Static/Static.cuh>

#include <Graph/GraphStd.hpp>
#include <Host/Classes/Timer.hpp>
#include <Device/Util/Timer.cuh>
#include "Util/CommandLineParam.hpp"

using namespace timer;
using namespace std::string_literals;
using namespace hornets_nest;

using vert_t = int;
using eoff_t = int;
using HornetGPU = hornet::gpu::Hornet<vert_t>;
using UpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
using Update = hornet::gpu::BatchUpdate<vert_t>;
using Init = hornet::HornetInit<vert_t>;
using hornet::SoAData;
using hornet::TypeList;
using hornet::DeviceType;

/**
 * @brief This test inserts and erases a number of batches
 */
int exec(int argc, char** argv) {
  
  if (argc != 4) {
    std::cout << "args: <graph filename> <batch size> <iterations count>\n";
    return -1;
  }
  
  int batch_size = std::stoi(argv[2]);
  int iterations = std::stoi(argv[3]);

  graph::GraphStd<vert_t, vert_t> loader;
  loader.read(argv[1]);

  Init initializer{loader.nV(), loader.nE(), loader.csr_out_offsets(), loader.csr_out_edges()};
  HornetGPU hornet{initializer};

  // Initial state of the graph
  auto initial_coo = hornet.getCOO();
  initial_coo.sort();
  
  std::multimap<vert_t, hornet::TypeList<vert_t>> initial_mmap = getHostMMap(initial_coo);
  std::cout << "Initial graph saved to CPU\n";
 
#if 1
  vert_t current_key = -1;
  for (auto it = initial_mmap.begin(); it != initial_mmap.end(); it++) {
    
    // Visit each key only once
    if (current_key == it->first) continue;
    else current_key = it->first;
    
    std::cout << current_key << " : ";
    auto key_values = initial_mmap.equal_range(current_key);
    for (auto val_it = key_values.first; val_it != key_values.second; val_it++)
      std::cout << std::get<0>(val_it->second) << " ";
    std::cout << "\n";
  }
#endif

  vert_t* src = new vert_t[batch_size];
  vert_t* dst = new vert_t[batch_size];
 
#define VERBOSE 0

  for(int i = 0; i < iterations; i++) {
    
    std::cout << "Generating batch of " << batch_size << " elements... ";
    generateBatch(loader, batch_size, src, dst, BatchGenType::INSERT);
    std::cout << "DONE\n";

    // The update in inserted into the graph and preprocessed
    Update update_insert{UpdatePtr{batch_size, src, dst}};
    update_insert.sort();

#if VERBOSE == 1
    std::cout << "Graph before update:\n";
    hornet.print();
#endif

    hornet.insert(update_insert, true, true);

    // Update was modified by the insert
    bool incomplete_update = update_insert.size() != batch_size;
    if (incomplete_update)
      std::cout << "Update was not completely inserted\n";

#if VERBOSE == 1
    update_insert.print();
    std::cout << "Graph after update:\n";
    hornet.print();
#endif

    // Do work in the middle
    auto after_insert_coo = hornet.getCOO();
    if (after_insert_coo.size() != initial_coo.size() + update_insert.size())
      std::cout << "Invalid mid size!\n";

    // Generate erase batch that removes the effective previous batch update
    auto soa_ptr = update_insert.in_edge().get_soa_ptr();
    UpdatePtr update_erase_ptr{
      update_insert.size(), soa_ptr.template get<0>(), soa_ptr.template get<1>()
    };

    // This should erase the insert update
    Update update_erase{update_erase_ptr};
    update_erase.sort();

#if VERBOSE == 1
    if (incomplete_update) {
      std::cout << "===\n";
      update_erase.print();
    }
#endif

    hornet.erase(update_erase);

#if VERBOSE == 1
    std::cout << "Graph after erase:\n";
    hornet.print();
#endif

    auto after_erase_coo = hornet.getCOO();
    if (after_erase_coo.size() != initial_coo.size()) {
      std::cout << "Invalid iteration size: " << after_erase_coo.size() << " != " << initial_coo.size() 
                << std::endl;
    }
  }

  delete[] src;
  delete[] dst;

  // Final state of the graph
  auto final_coo = hornet.getCOO();
  final_coo.sort();

#if 0
  // ======================================================================================
  // Check consistency of graph

  auto *s = host_initial_coo.srcPtr();
  auto *d = host_initial_coo.dstPtr();
    
  auto *S = host_final_coo.srcPtr();
  auto *D = host_final_coo.dstPtr();
    
  auto len = host_initial_coo.size();
  std::cout << "Edge count: " << len << std::endl;

  if (host_final_coo.size() != host_initial_coo.size()) {
    std::cout << "\nInit Size "<< host_initial_coo.size() << " != Combined size " << host_final_coo.size() << "\n";
  }

  bool valid = true;
  int counter = 0;
  for (int i = 0; i < len; i++) {
    bool error = (s[i] != S[i]) || (d[i] != D[i]);
    if (error) {
      valid = false;
      printf("%5d -> %5d | %5d -> %5d => %s\n",
        s[i], d[i], S[i], D[i], "ERROR");
    }
    counter += 1;
  }

  std::cout << "Effective size: " << counter << std::endl;
  std::cout << (valid ? "PASSED" : "NOT PASSED") 
            << std::endl;
#endif

  std::multimap<vert_t, hornet::TypeList<vert_t>> final_mmap = getHostMMap(final_coo);
  std::cout << "Final graph saved to CPU\n";
 
#if 1
  current_key = -1;
  for (auto it = final_mmap.begin(); it != final_mmap.end(); it++) {
    
    // Visit each key only once
    if (current_key == it->first) continue;
    else current_key = it->first;
    
    std::cout << current_key << " : ";
    auto key_values = final_mmap.equal_range(current_key);
    for (auto val_it = key_values.first; val_it != key_values.second; val_it++)
      std::cout << std::get<0>(val_it->second) << " ";
    std::cout << "\n";
  }
#endif

  std::cout << "Valid: " 
            << (final_mmap == initial_mmap ? "true" : "false")
            << std::endl;

  return 0;
}

int main(int argc, char* argv[]) {
  int ret = 0;
  {
    ret = exec(argc, argv);
  }

  return ret;
}

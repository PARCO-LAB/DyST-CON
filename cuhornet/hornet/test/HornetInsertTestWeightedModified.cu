#include "Hornet.hpp"
#include "StandardAPI.hpp"
#include "Util/BatchFunctions.hpp"
#include "Util/RandomGraphData.cuh"
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <Device/Util/CudaUtil.cuh>     //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include <cuda_profiler_api.h>

#include <Graph/GraphStd.hpp>
#include <Host/Classes/Timer.hpp>
#include <Device/Util/Timer.cuh>
#include "Util/CommandLineParam.hpp"

//using namespace hornets_nest;
using namespace timer;
using namespace std::string_literals;

using vert_t = int;
using eoff_t = int;
using wgt0_t = int;
using wgt1_t = float;
using Init = hornet::HornetInit<vert_t>;
using HornetGPU = hornet::gpu::Hornet<vert_t>;
using UpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
using Update = hornet::gpu::BatchUpdate<vert_t>;
using hornet::TypeList;
using hornet::DeviceType;

/**
 * @brief Example tester for Hornet
 */
int exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    int batch_size = std::stoi(argv[2]);
    graph::GraphStd<vert_t, vert_t> graph;
    graph.read(argv[1]);

    std::cout << "Initializing hornet ... ";
    Init hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    HornetGPU device_graph(hornet_init);
    std::cout << "done\n";

    std::cout << "Generating update batch (size: " << batch_size << ") ... ";
    hornet::RandomGenTraits<hornet::EMPTY> cooGenTraits;
    auto randomBatch = hornet::generateRandomCOO<vert_t, eoff_t>(graph.nV(), batch_size, cooGenTraits);
    std::cout << "done\n";

    std::cout << "Batch edges:\n";
    Update batch_update(randomBatch);
    batch_update.print();

    std::cout << "Applying batch to host COO ... ";
    auto base_coo = device_graph.getCOO(true);
    base_coo.append(randomBatch);
    base_coo.sort();
    std::cout << "done\n";

    std::cout << "Applying batch to device graph and retriving COO ... ";
    device_graph.insert(batch_update);
    auto device_coo = device_graph.getCOO(true);
    device_coo.sort();
    std::cout << "done\n";

    std::cout << "Creating multimaps for testing correctness ... ";
    std::multimap<vert_t, TypeList<vert_t>> base_coo_map   = getHostMMap(base_coo);
    std::multimap<vert_t, TypeList<vert_t>> device_coo_map = getHostMMap(device_coo);
    std::cout << "done\n";

#if 0
    std::cout << "Base COO:\n"; 
    vert_t current_key = -1;
    for (auto it = base_coo_map.begin(); it != base_coo_map.end(); it++) {
    
      // Visit each key only once
      if (current_key == it->first) continue;
      else current_key = it->first;
    
      std::cout << current_key << " : ";
      auto key_values = base_coo_map.equal_range(current_key);
      for (auto val_it = key_values.begin(); val_it != key_values.end(); val_it++)
        std::cout << val_it->second << "\n";
    }

    std::cout << "Device COO:\n"; 
    current_key = -1;
    for (auto it = device_coo_map.begin(); it != device_coo_map.end(); it++) {
    
      // Visit each key only once
      if (current_key == it->first) continue;
      else current_key = it->first;
    
      std::cout << current_key << " : ";
      auto key_values = device_coo_map.equal_range(current_key);
      for (auto val_it = key_values.begin(); val_it != key_values.end(); val_it++)
        std::cout << val_it->second << "\n";
    }
#endif

    std::cout << "Initial size: "  << base_coo_map.size() 
              << ", Device size: " << device_coo_map.size() 
              << "\n";

#if 0
    auto base_it   = base_coo_map.begin();
    auto device_it = device_coo_map.begin();
    
    while (init_it != init_coo_map.end() && inst_it != inst_coo_map.end()) {

       

      init_it++;
      inst_it++;
    }
#else
    std::cout<<"...Done!\n";
    if (base_coo_map == device_coo_map) {
      std::cout<<"Passed\n";
    } else {
      std::cout<<"Failed\n";
    }
#endif

    return 0;
}

int main(int argc, char* argv[]) {
  int ret = 0;
  {
    ret = exec(argc, argv);
  }

  return ret;
}

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

//using namespace hornets_nest;
using namespace timer;
using namespace std::string_literals;

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
 * @brief Example tester for Hornet
 */
int exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    graph::GraphStd<vert_t, vert_t> graph;
    graph.read(argv[1]);
    int batch_size = std::stoi(argv[2]);
    Init hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());

    HornetGPU hornet_gpu(hornet_init);
    auto init_coo = hornet_gpu.getCOO();

    hornet::RandomGenTraits<hornet::EMPTY> cooGenTraits(true, false, 100);
    auto randomBatch = hornet::generateRandomCOO<vert_t, eoff_t>(graph.nV(), batch_size, cooGenTraits);
    Update batch_update(randomBatch);

#if 0
    printf("Generated batch: \n");
    batch_update.print();

    printf("Graph before update:\n");
    hornet_gpu.print();
#endif

    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";
    Timer<DEVICE> TM(3);
    TM.start();
    hornet_gpu.insert(batch_update);
    TM.stop();

    printf("ne: %d\n", hornet_gpu.nE());
    std::cout<<"=======\n";
    TM.print("Insertion " + std::to_string(batch_size) + ":  ");

#if 0
    printf("Graph after update:\n");
    hornet_gpu.print();
#endif

    auto inst_coo = hornet_gpu.getCOO();
    init_coo.append(randomBatch);
    init_coo.sort();
    inst_coo.sort();

#if 1
    hornet::COO<DeviceType::HOST, vert_t, hornet::EMPTY, eoff_t> host_init_coo = init_coo;
    hornet::COO<DeviceType::HOST, vert_t, hornet::EMPTY, eoff_t> host_inst_coo = inst_coo;

    auto *s = host_init_coo.srcPtr();
    auto *d = host_init_coo.dstPtr();
    
    auto *S = host_inst_coo.srcPtr();
    auto *D = host_inst_coo.dstPtr();
    
    auto len = host_init_coo.size();
    std::cout << "Edge count: " << len << std::endl;

    if (host_inst_coo.size() != host_init_coo.size()) {
      std::cout << "\nInit Size "<< host_init_coo.size() << " != Combined size " << host_inst_coo.size() << "\n";
      return -1;
    }

    bool valid = true;
    int counter = 0;
    for (int i = 0; i < len; i++) {
      bool error = (s[i] != S[i]) || (d[i] != D[i]); 
      if (error) valid = false;

#if 0
      printf("%5d -> %5d | %5d -> %5d => %s\n",
          s[i], d[i], S[i], D[i], error ? "ERROR" : "OK");
#endif

      counter += 1;
    }

    std::cout << "Effective size: " << counter << std::endl;
    std::cout << (valid ? "PASSED" : "NOT PASSED") 
              << std::endl;

#else
    using mapT = std::multimap<vert_t, TypeList<vert_t>>;
    
    std::cout<<"Creating multimap for testing correctness...";
    mapT init_coo_map = getHostMMap(init_coo);
    mapT inst_coo_map = getHostMMap(inst_coo);
    std::cout<<"...Done!\n";

    std::set<mapT::value_type> init_set(init_coo_map.begin(), init_coo_map.end());
    std::set<mapT::value_type> inst_set(inst_coo_map.begin(), inst_coo_map.end());
    
    mapT result;
    std::set_symmetric_difference(init_set.begin(), init_set.end(),
                                  inst_set.begin(), inst_set.end(),
                                  std::inserter(result, result.end()));
    
    if (result.size() == 0)
      std::cout << "PASSED\n";
    else {

      std::cout << "NOT PASSED, symmetric_difference:\n";
      for (auto it = result.begin(); it != result.end(); it++) {
        std::cout << it->first << " -> " << std::get<0>(it->second) << "\n";
      }

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

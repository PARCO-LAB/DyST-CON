/**
 * @file STCONTest.cu
 * @brief Test of Dynamic ST-CON
 */

#include <cmath>
#include <iostream>

#include <Graph/GraphStd.hpp>
#include <StandardAPI.hpp>
#include <Util/BatchFunctions.hpp>
#include <Util/CommandLineParam.hpp>

#include "Dynamic/ST-CON/STCON.cuh"
#include "Static/ClusteringCoefficient/cc.cuh"

namespace test {

using namespace hornets_nest;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;

using HornetInit = ::hornet::HornetInit<vert_t>;
using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
using HornetBatchUpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
using HornetBatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

void generateBatch(vert_t *src, vert_t *dst, int batch_size, int dist_size) {
    // srand(time(0));
    for (int i = 0; i < batch_size; i++) {

        vert_t src_id = rand() % dist_size;
        vert_t dst_id = rand() % dist_size;

        src[i] = src_id;
        dst[i] = dst_id;
    }
}

int exec(int argc, char **argv) {
    graph::GraphStd<vert_t, vert_t> host_graph(ENABLE_INGOING);
    host_graph.read(argv[1], PRINT_INFO);
    int batch_size = std::stoi(argv[2]);

    HornetInit graph_init{host_graph.nV(), host_graph.nE(),
                          host_graph.csr_out_offsets(),
                          host_graph.csr_out_edges()};

    HornetGraph device_graph{graph_init};
    srand(time(NULL));
    int source = rand() % device_graph.nV();
    int target = rand() % device_graph.nV();
    // Test on small.mtx
    // int source = 1;
    // int target = 39;
    std::cout << "s = " << source << ", t = " << target << std::endl;
    STCON<HornetGraph> stcon{device_graph, source, target};

    timer::Timer<timer::DEVICE> timer;

    stcon.reset();
    timer.start();
    stcon.run();
    timer.stop();
    auto originalDuration = timer.duration();

    auto stats = stcon.get_stats();
    // unsigned long long int nMemoryAccesses = stats.nMemoryAccesses;
    std::cout << "---------FIRST RUN---------"
              << "\nConnection found: " << stats.connectionFound
              << "\nElapsed time: " << originalDuration
            //   << "\nNumber of memory accesses: " << nMemoryAccesses
              << std::endl << std::endl;

    // =======================================================================
    // BATCH GENERATION AND INSERTION

    // Double the batch size if the graph is undirected (an edge from src to dest and another edge from dest to src)
    int allocated_batch_size = host_graph.is_undirected() ? batch_size * 2 : batch_size;

    vert_t *batch_src = new vert_t[allocated_batch_size];
    vert_t *batch_dst = new vert_t[allocated_batch_size];

    printf("Generating batch of %d edges\n", allocated_batch_size);

    generateBatch(batch_src, batch_dst, batch_size, device_graph.nV());

    if (host_graph.is_undirected()) {
        memcpy(batch_src + batch_size, batch_dst, sizeof(vert_t) * batch_size);
        memcpy(batch_dst + batch_size, batch_src, sizeof(vert_t) * batch_size);
    }

    HornetBatchUpdatePtr update_data_ptr{allocated_batch_size, batch_src, batch_dst};
    HornetBatchUpdate update{update_data_ptr};

#if 0
    printf("=====================================\n");
    printf("Initial update batch (insert):\n");
    update.print();
    // printf("Graph before and after batch\n");
    // device_graph.print();
    device_graph.insert(update, true, true);
    // printf(" --- \n");
    device_graph.print();
#else
    device_graph.insert(update, true, true);
#endif

    // =======================================================================
    // CHECK CONNECTION STATUS ON THE UPDATED GRAPH

    // Insert edges
    stcon.update(update, true);

    stats = stcon.get_stats();
    // nMemoryAccesses = stats.nMemoryAccesses;
    std::cout << "---------UPDATE (INSERT)---------"
              << "\nConnection found: " << stats.connectionFound
              << "\nvertex update time: " << stats.vertex_update_time
              << "\nexpansion time: " << stats.expansion_time
              << "\ntotal time: " << stats.vertex_update_time + stats.expansion_time
            //   << "\nNumber of memory accesses: " << nMemoryAccesses
              << std::endl << std::endl;
              
    timer.start();
    stcon.reset();
    stcon.run();
    timer.stop();
    originalDuration = timer.duration();

    stats = stcon.get_stats();
    // nMemoryAccesses = stats.nMemoryAccesses + device_graph.nV();
    std::cout << "---------SECOND RUN---------"
              << "\nConnection found: " << stats.connectionFound
              << "\nElapsed time: " << originalDuration
            //   << "\nNumber of memory accesses: " << nMemoryAccesses
              << "\nSpeed up: " << timer.duration() / (stats.vertex_update_time + stats.expansion_time)
              << "\nTime elapsed: " << timer.duration()
              << std::endl << std::endl;


    // timer.start();
    // stcon.reset();
    // stcon.run();
    // timer.stop();
    // stats = stcon.get_stats();
    // std::cout << "Validation: Connection found: " << stats.connectionFound << std::endl;
    // std::cout << "Speed up: " << timer.duration() / (stats.vertex_update_time + stats.expansion_time) << std::endl;
    // std::cout << timer.duration() << std::endl << std::endl;

    HornetBatchUpdatePtr update_data_ptr2{allocated_batch_size, batch_src, batch_dst};
    HornetBatchUpdate update2{update_data_ptr2};
#if 0
    printf("=====================================\n");
    printf("Initial update batch (erase):\n");
    update2.print();
    printf("Graph before and after batch\n");
    device_graph.print();
    device_graph.erase(update2, true);
    printf(" --- \n");
    device_graph.print();
#else
    device_graph.erase(update2, true);
#endif

    // Remove edges
    stcon.update(update2, false);

    stats = stcon.get_stats();
    // nMemoryAccesses = stats.nMemoryAccesses;
    std::cout << "---------UPDATE (ERASE)---------"
              << "\nConnection found: " << stats.connectionFound
              << "\nvertex update time: " << stats.vertex_update_time
              << "\nexpansion time: " << stats.expansion_time
              << "\ntotal time: " << stats.vertex_update_time + stats.expansion_time
            //   << "\nNumber of memory accesses: " << nMemoryAccesses
              << std::endl << std::endl;

    timer.start();
    stcon.reset();
    stcon.run();
    timer.stop();
    originalDuration = timer.duration();
    stats = stcon.get_stats();
    // nMemoryAccesses = stats.nMemoryAccesses + device_graph.nV();
    std::cout << "---------SECOND RUN---------"
              << "\nConnection found: " << stats.connectionFound
              << "\nElapsed time: " << originalDuration
            //   << "\nNumber of memory accesses: " << nMemoryAccesses
              << "\nSpeed up: " << originalDuration / (stats.vertex_update_time + stats.expansion_time)
              << std::endl << std::endl;
    // std::cout << "Validation: Connection found: " << stats.connectionFound << std::endl;
    // std::cout << "Number of memory accesses: " << nMemoryAccesses << std::endl;
    // std::cout << "Speed up: " << originalDuration / (stats.vertex_update_time + stats.expansion_time) << std::endl << std::endl;

    // stcon.validate();

//     timer.start();
//     stcon.reset();
//     stcon.run();
//     timer.stop();
//     originalDuration = timer.duration();

//     stats = stcon.get_stats();
//     nMemoryAccesses = stats.nMemoryAccesses;
//     std::cout << "---------FIRST RUN---------"
//               << "\nConnection found: " << stats.connectionFound
//               << "\nElapsed time: " << originalDuration
//               << "\nNumber of memory accesses: " << nMemoryAccesses << std::endl << std::endl;

    return 0;
}

} // namespace test

int main(int argc, char **argv) {
    int ret = 0;
    {
        ret = test::exec(argc, argv);
    }
    return ret;
}

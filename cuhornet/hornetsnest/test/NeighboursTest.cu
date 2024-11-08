/**
 * @file DBFS.cu
 * @brief Test of Dynamic BFS
*/

#include <iostream>

#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <Util/BatchFunctions.hpp>

#include "Dynamic/DynamicBFS/DBFS.cuh"

namespace test {

    using namespace hornets_nest;
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    using HornetInit = ::hornet::HornetInit<vert_t>;
    using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
    using HornetBatchUpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
    using HornetBatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

    int exec(int argc, char** argv) {

        graph::GraphStd<vert_t, vert_t> host_graph;
        host_graph.read(argv[1], PRINT_INFO | SORT);

        HornetInit graph_init{host_graph.nV(),
                              host_graph.nE(),
                              host_graph.csr_out_offsets(),
                              host_graph.csr_out_edges()};

        HornetInit graph_init_inv{host_graph.nV(),
                                  host_graph.nE(),
                                  host_graph.csr_in_offsets(),
                                  host_graph.csr_in_edges()};

        printf(" ============== Graph ================\n");
        HornetGraph device_graph{graph_init};
        device_graph.print();

        printf(" ============== Inverse Graph ========\n");
        HornetGraph device_graph_inv{graph_init_inv};
        device_graph_inv.print();

#if 0
        DynamicBFS<HornetGraph> DBFS{device_graph, device_graph_inv};

        DBFS.print_nodes();
        DBFS.print_children(1);
        DBFS.print_parents(1);
#endif

        vert_t batch_src[] = { 8  };
        vert_t batch_dst[] = { 11 };

        HornetBatchUpdatePtr update_src_dst_ptr{1, batch_src, batch_dst};
        HornetBatchUpdate update_src_dst{update_src_dst_ptr};
        device_graph.insert(update_src_dst, true, true);

        HornetBatchUpdatePtr update_dst_src_ptr{1, batch_dst, batch_src};
        HornetBatchUpdate update_dst_src{update_dst_src_ptr};
        device_graph_inv.insert(update_dst_src, true, true);

        printf(" ============== Graph ================\n");
        device_graph.print();

        printf(" ============== Inverse Graph ========\n");
        device_graph_inv.print();

        return 0;
    }

} // namespace test

int main(int argc, char** argv) {
    int ret = 0;
    {
        // ?
        ret = test::exec(argc, argv);
    }
    return ret;
}
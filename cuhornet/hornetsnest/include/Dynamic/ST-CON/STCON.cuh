#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>
#include <Device/Util/Timer.cuh>
#include <LoadBalancing/LogarithmRadixBinning.cuh>
#include <Macros.hpp>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <algorithm>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <iterator>
#include <stdio.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

namespace hornets_nest {

using vert_t = int;
using dist_t = int;
using color_t = int;

using BatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

template <typename HornetGraph>
class STCON : public StaticAlgorithm<HornetGraph> {
public:
    struct Stats {
        // Time measures for dynamic update
        float vertex_update_time{0};
        float expansion_time{0};

        // unsigned long long nMemoryAccesses{0};
        bool connectionFound{false};
    };

public:
    STCON(HornetGraph &graph, vert_t source, vert_t target);
    virtual ~STCON();

    void reset() override;
    void run() override;
    void release() override;
    bool validate() override;

    Stats get_stats() const;

    void update(BatchUpdate &batch, const bool is_insert);

private:
    HornetGraph &_graph;

    BufferPool _buffer_pool;
    load_balancing::BinarySearch _load_balancing;
    TwoLevelQueue<vert_t> _frontier;

    Stats _stats{};

    // Used to measure algorithm performance
    timer::Timer<timer::DEVICE> _device_timer;

    // Nodes colors
    color_t *_colors{nullptr};

    // Source and target nodes
    vert_t _source{0};
    vert_t _target{0};

    void visit(bool *devConnectionFound/*, unsigned long long *nMemoryAccesses*/);

    void debugConnection();
};

} // namespace hornets_nest

namespace hornets_nest {

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//                          Algorithm Operators
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

struct Expand {
    color_t *d_colors;
    TwoLevelQueue<vert_t> queue;
    bool *connectionFound;
    // unsigned long long *nMemoryAccesses;

    OPERATOR(Vertex &vertex, Edge &edge) {
        // printf("Expanding %d -> %d\n", vertex.id(), edge.dst_id());
        auto dst = edge.dst_id();

        // Double-Source version
        if (atomicCAS(d_colors + dst, INF, *(d_colors + vertex.id())) == INF) {
            queue.insert(dst);
            // atomicAdd(nMemoryAccesses, 3);
        }
        else if (d_colors[dst] != d_colors[vertex.id()]) {
            // printf("Connection found, node %d\n", dst);
            *connectionFound = true;
        }
    }
};

struct BatchVertexUpdate {
    color_t *colors;
    TwoLevelQueue<vert_t> frontier;
    bool *connectionFound;
    bool *mustChangeConnectionStatus;
    bool *twoNodesSameColor;
    bool *colorsRemoved;
    bool is_insert;
    // unsigned long long *nMemoryAccesses;

    OPERATOR(Vertex &src, Vertex &dst) {
        if (*connectionFound && is_insert)
            return;

        if (!(*connectionFound) && !is_insert)
            return;
            
        const color_t src_color = colors[src.id()];
        const color_t dst_color = colors[dst.id()];
        // atomicAdd(nMemoryAccesses, 2);

        if (*connectionFound) {
            if (is_insert) {
                // If we assume only one update, no operations are required.
                // If we assume multiple updates, the frontier needs to be updated
                // Connection found and edges inserted
                if (src_color != INF && dst_color == INF) {
                    // printf("src %d (color %d) inserts dst %d (color %d)\n", src.id(), src_color, dst.id(), dst_color);
                    colors[dst.id()] = src_color;
                    frontier.insert(dst.id());
                    // atomicAdd(nMemoryAccesses, 2);
                }
            }

            if (!is_insert) {
                // Connection found and edges removed
                if (src_color == dst_color && src_color != INF) {
                    // Removing an edge between two nodes of the same color
                    // Must reset all the nodes of that color to start a new BFS
                    *twoNodesSameColor = true;
                    *mustChangeConnectionStatus = true;

                    if (src_color == 1 || dst_color == 1)
                        colorsRemoved[0] = true;
                    if (src_color == 2 || dst_color == 2)
                        colorsRemoved[1] = true;
                }
                else if (src_color != dst_color && src_color != INF && dst_color != INF) {
                    // Removing an edge between two nodes of different colors
                    *mustChangeConnectionStatus = true;
                    // printf("different colors\n");
                }

                // printf("src %d, color %d - dst %d, color %d\n", src.id(), src_color, dst.id(), dst_color);
            }
        }
        else {
            if (is_insert) {
                // Connection not found and edges inserted
                if (src_color != INF && dst_color == INF) {
                    // printf("src %d (color %d) inserts dst %d (color %d)\n", src.id(), src_color, dst.id(), dst_color);
                    colors[dst.id()] = src_color;
                    frontier.insert(dst.id());
                    // atomicAdd(nMemoryAccesses, 2);
                }
                else if (src_color != INF && dst_color != INF && src_color != dst_color) {
                    // printf("Different colored nodes connected\n");
                    *mustChangeConnectionStatus = true;
                }
            }
            else {
                // Connection not found and edges removed

            }
        }

        // printf("src %d, color %d - dst %d, color %d\n", src.id(), src_color, dst.id(), dst_color);
    }
};

//------------------------------------------------------------------------------
template <typename HornetGraph>
STCON<HornetGraph>::STCON(HornetGraph &graph, vert_t source, vert_t target)
    : StaticAlgorithm<HornetGraph>{graph}, _load_balancing{graph}, _frontier{graph, 10.0f}, _graph{graph}, _source{source}, _target{target} {
    _buffer_pool.allocate(&_colors, _graph.nV());
    reset();
}

template <typename HornetGraph>
STCON<HornetGraph>::~STCON() {
    release();
}

template <typename HornetGraph>
void STCON<HornetGraph>::reset() {
    _stats.connectionFound = false;
    _frontier.clear();
    // _stats.nMemoryAccesses = 0;

    color_t *colors = _colors;
    vert_t *source, *target;
    cudaMalloc(&source, sizeof(vert_t));
    cudaMalloc(&target, sizeof(vert_t));
    cudaMemcpy(source, &_source, sizeof(vert_t), cudaMemcpyHostToDevice);
    cudaMemcpy(target, &_target, sizeof(vert_t), cudaMemcpyHostToDevice);
    forAllnumV(StaticAlgorithm<HornetGraph>::hornet,
                [=] __device__(int i) {
                    if (i == *source)
                        colors[i] = 1;
                    else if (i == *target)
                        colors[i] = 2;
                    else
                        colors[i] = INF;
                });
    cudaFree(source);
    cudaFree(target);

    // debugConnection();
}

template <typename HornetGraph>
void STCON<HornetGraph>::run() {
    _frontier.insert(_source);
    _frontier.insert(_target);

    bool *devConnectionFound;
    cudaMalloc(&devConnectionFound, sizeof(bool));
    cudaMemset(devConnectionFound, false, sizeof(bool));
    // unsigned long long *nMemoryAccesses;
    // cudaMalloc(&nMemoryAccesses, sizeof(unsigned long long));
    // cudaMemset(nMemoryAccesses, _stats.nMemoryAccesses, sizeof(int));

    visit(devConnectionFound/*, nMemoryAccesses*/);

    cudaFree(devConnectionFound);

    // debugConnection();
}

template <typename HornetGraph>
void STCON<HornetGraph>::release() {
    gpu::free(_colors, _graph.nV());
    _colors = nullptr;
}

template <typename HornetGraph>
bool STCON<HornetGraph>::validate() {
    printf("Validating...\n");
    reset();
    run();

    // TODO: check if the result is the same
    return true;
}

template <typename HornetGraph>
auto STCON<HornetGraph>::get_stats() const -> Stats {
    return _stats;
}

template <typename HornetGraph>
void STCON<HornetGraph>::update(BatchUpdate &batch, const bool is_insert) {
    _stats.vertex_update_time = 0;
    _stats.expansion_time = 0;
    // _stats.nMemoryAccesses = 0;

    bool *devConnectionFound;
    bool mustChangeConnectionStatus, *devMustChangeConnectionStatus; // Used to avoid race conditions
    bool twoNodesSameColor, *devTwoNodesSameColor;
    bool colorsRemoved[2] = {false, false};
    bool *devColorsRemoved;
    cudaMalloc(&devConnectionFound, sizeof(bool));
    cudaMemset(devConnectionFound, _stats.connectionFound, sizeof(bool));
    cudaMalloc(&devMustChangeConnectionStatus, sizeof(bool));
    cudaMemset(devMustChangeConnectionStatus, false, sizeof(bool));
    cudaMalloc(&devTwoNodesSameColor, sizeof(bool));
    cudaMemset(devTwoNodesSameColor, false, sizeof(bool));
    cudaMalloc(&devColorsRemoved, 2 * sizeof(bool));
    cudaMemcpy(devColorsRemoved, colorsRemoved, 2 * sizeof(bool), cudaMemcpyHostToDevice);

    // unsigned long long *nMemoryAccesses;
    // cudaMalloc(&nMemoryAccesses, sizeof(int));
    // cudaMemset(nMemoryAccesses, _stats.nMemoryAccesses, sizeof(int));
    
    _device_timer.start();
    forAllEdgesBatch(_graph,
                     batch,
                     BatchVertexUpdate{_colors,
                                       _frontier,
                                       devConnectionFound,
                                       devMustChangeConnectionStatus,
                                       devTwoNodesSameColor,
                                       devColorsRemoved,
                                       is_insert,
                                    //    nMemoryAccesses
                                       }
                    );
    _device_timer.stop();
    _stats.vertex_update_time = _device_timer.duration();
    // cudaMemcpy(&_stats.nMemoryAccesses, nMemoryAccesses, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(&mustChangeConnectionStatus, devMustChangeConnectionStatus, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("mustChangeConnectionStatus: %d\n", mustChangeConnectionStatus);
    if (_stats.connectionFound && !mustChangeConnectionStatus) {
        printf("Connection already found during the first visit\n");
        return;
    }
    if (!_stats.connectionFound && !mustChangeConnectionStatus && !is_insert) {
        printf("Connection cannot be found\n");
        color_t *colors = _colors;
        vert_t *source, *target;
        // unsigned long long *dev_memoryAccesses;
        // const unsigned long long ZERO = 0;
        cudaMalloc(&source, sizeof(vert_t));
        cudaMalloc(&target, sizeof(vert_t));
        // cudaMalloc(&dev_memoryAccesses, sizeof(unsigned long long));
        cudaMemcpy(source, &_source, sizeof(vert_t), cudaMemcpyHostToDevice);
        cudaMemcpy(target, &_target, sizeof(vert_t), cudaMemcpyHostToDevice);
        // cudaMemcpy(dev_memoryAccesses, &ZERO, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        forAllnumV(StaticAlgorithm<HornetGraph>::hornet,
                    [=] __device__(int i) {
                        // if (i != *source && i != *target)
                        //     colors[i] = INF;
                        if ((i != *source && i != *target) &&
                            ((colorsRemoved[0] && colors[i] == colors[*source]) ||
                             (colorsRemoved[1] && colors[i] == colors[*target]))) {
                                colors[i] = INF;
                                // Increment memory access
                                // atomicAdd(dev_memoryAccesses, 2);
                                // atomicAdd(nMemoryAccesses, 2);
                        }
                    });
        cudaFree(source);
        cudaFree(target);
        // cudaMemcpy(&_stats.nMemoryAccesses, nMemoryAccesses, sizeof(int), cudaMemcpyDeviceToHost);
        return;
    }
    
    // Check if a connection has been found
    if (mustChangeConnectionStatus) {
        _stats.connectionFound = !_stats.connectionFound;
        cudaMemset(devConnectionFound, _stats.connectionFound, sizeof(bool));
    }
    if (_stats.connectionFound) {
        printf("Connection found when inserting the edges\n");
        return;
    }

    _device_timer.start();
    cudaMemcpy(&twoNodesSameColor, devTwoNodesSameColor, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&colorsRemoved, devColorsRemoved, 2 * sizeof(bool), cudaMemcpyDeviceToHost);
    // Possible connection removed when removing an edge (edge between two nodes of the same color or two nodes of different color)
    // Must restart the visit from scratch to check if there is still a connection
    if (!_stats.connectionFound && !is_insert && mustChangeConnectionStatus && twoNodesSameColor) {
        _frontier.clear();
        if (colorsRemoved[0])
            _frontier.insert(_source);
        if (colorsRemoved[1])
            _frontier.insert(_target);
        // _stats.nMemoryAccesses += _frontier.size();

        color_t *colors = _colors;
        vert_t *source, *target;
        // unsigned long long *dev_memoryAccesses;
        // const unsigned long long ZERO = 0;
        cudaMalloc(&source, sizeof(vert_t));
        cudaMalloc(&target, sizeof(vert_t));
        // cudaMalloc(&dev_memoryAccesses, sizeof(unsigned long long));
        cudaMemcpy(source, &_source, sizeof(vert_t), cudaMemcpyHostToDevice);
        cudaMemcpy(target, &_target, sizeof(vert_t), cudaMemcpyHostToDevice);
        // cudaMemcpy(dev_memoryAccesses, &ZERO, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        forAllnumV(StaticAlgorithm<HornetGraph>::hornet,
                    [=] __device__(int i) {
                        // if (i != *source && i != *target)
                        //     colors[i] = INF;
                        if ((i != *source && i != *target) &&
                            ((colorsRemoved[0] && colors[i] == colors[*source]) ||
                             (colorsRemoved[1] && colors[i] == colors[*target]))) {
                                colors[i] = INF;
                                // Increment memory access
                                // atomicAdd(dev_memoryAccesses, 2);
                                // atomicAdd(nMemoryAccesses, 2);
                        }
                    });
        cudaFree(source);
        cudaFree(target);
        // cudaMemcpy(&_stats.nMemoryAccesses, nMemoryAccesses, sizeof(int), cudaMemcpyDeviceToHost);

        printf("frontier size: %d\n", _frontier.size());

        // debugConnection();
    }
    else if (!_stats.connectionFound && !is_insert && mustChangeConnectionStatus) { // Edge removed between two nodes of different colors
        printf("ELSE IF\n");
        printf("frontier size: %d\n", _frontier.size());
        // _frontier.swap();
        printf("frontier size: %d\n", _frontier.size());
    }
    else {
        printf("ELSE\n");
        printf("frontier size: %d\n", _frontier.size());
        _frontier.swap();
        printf("frontier size: %d\n", _frontier.size());
    }

    // _frontier.print();
    // _frontier.print_output();

    // _device_timer.start();
    visit(devConnectionFound/*, nMemoryAccesses*/);
    _device_timer.stop();
    _stats.expansion_time = _device_timer.duration();

    if (_stats.connectionFound) {
        printf("Connection found after continuing the bfs\n");
    }

    cudaFree(devConnectionFound);
    cudaFree(devMustChangeConnectionStatus);
    cudaFree(devTwoNodesSameColor);
    cudaFree(devColorsRemoved);

    // debugConnection();
}

template <typename HornetGraph>
void STCON<HornetGraph>::visit(bool *devConnectionFound/*, unsigned long long *nMemoryAccesses*/) {
    while (_frontier.size() != 0 && !_stats.connectionFound) {
        forAllEdges(StaticAlgorithm<HornetGraph>::hornet, _frontier,
                    Expand{_colors, _frontier, devConnectionFound, /*nMemoryAccesses*/},
                    _load_balancing);
        
        _frontier.swap();

        cudaMemcpy(&_stats.connectionFound, devConnectionFound, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // cudaMemcpy(&_stats.nMemoryAccesses, nMemoryAccesses, sizeof(int), cudaMemcpyDeviceToHost);
}

template <typename HornetGraph>
void STCON<HornetGraph>::debugConnection() {
    // Print color vector
    cudaDeviceSynchronize();
    auto* colors = new color_t[_graph.nV()];
    gpu::copyToHost(_colors, _graph.nV(), colors);
    for (int i = 0; i < _graph.nV(); i++)
        if (colors[i] != INF)
            printf("[ST-CON] node: %d | color: %d\n", i, colors[i]);
    delete[] colors;
}

} // namespace hornets_nest

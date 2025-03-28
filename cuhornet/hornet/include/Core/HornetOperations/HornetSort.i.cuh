/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <limits>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>
#include <thrust/transform.h>

namespace hornet {

namespace gpu {

// LRB segmented sort is based on the following papers:
// Fox, James, Alok Tripathy, and Oded Green. "Improving Scheduling for
// Irregular Applications with Logarithmic Radix Binning." IEEE High Performance
// Extreme Computing Conference (HPEC), 2019 Green, Oded, James Fox, Alex
// Watkins, Alok Tripathy, Kasimir Gabert, Euna Kim, Xiaojing An, Kumar Aatish,
// and David A. Bader. "Logarithmic radix binning and vectorized triangle
// counting." IEEE High Performance extreme Computing Conference (HPEC), 2018.

template <typename degree_t> struct InvalidEdgeCount {
  __device__ degree_t operator()(degree_t deg) {
    if (deg == 0)
      return 0;
    else
      return (1 << xlib::ceil_log2(deg)) - deg;
  }
};

template <int BLOCK_SIZE, typename HornetDeviceT, typename vid_t,
          typename degree_t>
__global__ void invalidateEdges(HornetDeviceT hornet, degree_t *offsets,
                                size_t offsets_count, vid_t max_vertex) {
  const int ITEMS_PER_BLOCK = xlib::smem_per_block<degree_t, BLOCK_SIZE>();
  __shared__ degree_t smem[ITEMS_PER_BLOCK];
  const auto &lambda = [&](int pos, degree_t edge_offset) {
    auto vertex = hornet.vertex(pos);
    vid_t *dst = vertex.neighbor_ptr() + vertex.degree();
    dst[edge_offset] = max_vertex;
  };

  xlib::simpleBinarySearchLB<BLOCK_SIZE>(offsets, offsets_count, smem, lambda);
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
          typename vid_t, typename degree_t>
void HORNET::sort(void) {
  if (_nE == 0) {
    return;
  }
  cudaStream_t stream{nullptr};

  rmm::device_vector<degree_t> offsets(_nV + 1);
  degree_t *vertex_degrees = _vertex_data.get_soa_ptr().template get<0>();
  thrust::transform(rmm::exec_policy(stream), vertex_degrees,
                    vertex_degrees + _nV, offsets.begin(),
                    InvalidEdgeCount<degree_t>());
  CHECK_CUDA_ERROR

  thrust::exclusive_scan(rmm::exec_policy(stream), offsets.begin(),
                         offsets.end(), offsets.begin());
  CHECK_CUDA_ERROR

  degree_t number_of_edges = offsets[_nV];
  HornetDeviceT hornet_device = device();
  const int BLOCK_SIZE = 256;
  int num_blocks = xlib::ceil_div(_nE, BLOCK_SIZE);
  if (num_blocks == 0) {
    return;
  }
  vid_t max = std::numeric_limits<vid_t>::max();
  invalidateEdges<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
      hornet_device, offsets.data().get(), offsets.size(), max);
  CHECK_CUDA_ERROR

  _ba_manager.sort();
  CHECK_CUDA_ERROR
}

#define RELABEL_VERBOSE 0
#define RELABEL_BLOCK_SIZE 1024
#define RELABEL_BLOCK_WORK 20
#define RELABEL_WORK (RELABEL_BLOCK_SIZE * RELABEL_BLOCK_WORK)

template <typename target_t>
__global__ void kernel_apply_relabeling(const int *relabeling,
                                        const target_t *target,
                                        target_t *output, const int N) {

  int grid_size = blockDim.x * gridDim.x;
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = global_idx; i < N; i += grid_size) {
    output[relabeling[i]] = target[i];
  }
}

/*
template <unsigned N, unsigned SIZE, typename... VertexMetaTypes>
void permute_type_array(SoAPtr<degree_t, xlib::byte_t *, degree_t, degree_t,
                               VertexMetaTypes...> &old_soa,
                        const int blocks_count) {
  if constexpr (N != SIZE) {
    kernel_apply_relabeling<<<blocks_count, RELABEL_BLOCK_SIZE>>>();
    permute_type_array<N + 1, SIZE, VertexMetaTypes...>(old_soa, blocks_count);
  }
}
*/

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
          typename vid_t, typename degree_t>
void HORNET::relabel(int *relabeling, bool sort_edges) {
  assert(sizeof...(VertexMetaTypes) == 0 &&
         "Relabeling does not support metatypes");

  // Where to store the new vertex data
  SoAData<TypeList<degree_t, xlib::byte_t *, degree_t, degree_t,
                   VertexMetaTypes...>,
          DeviceType::DEVICE>
      new_vertex_data(nV());

  auto new_soa = new_vertex_data.get_soa_ptr();
  auto old_soa = _vertex_data.get_soa_ptr();

  const int blocks_count = (nV() + RELABEL_WORK - 1) / RELABEL_WORK;

  // Vertex degree
  kernel_apply_relabeling<degree_t><<<blocks_count, RELABEL_BLOCK_SIZE>>>(
      relabeling, old_soa.template get<0>(), new_soa.template get<0>(), nV());
  // Pointer to vertex block array
  kernel_apply_relabeling<xlib::byte_t *><<<blocks_count, RELABEL_BLOCK_SIZE>>>(
      relabeling, old_soa.template get<1>(), new_soa.template get<1>(), nV());
  // Block index in block array
  kernel_apply_relabeling<degree_t><<<blocks_count, RELABEL_BLOCK_SIZE>>>(
      relabeling, old_soa.template get<2>(), new_soa.template get<2>(), nV());
  // Edges count
  kernel_apply_relabeling<degree_t><<<blocks_count, RELABEL_BLOCK_SIZE>>>(
      relabeling, old_soa.template get<3>(), new_soa.template get<3>(), nV());

  /*
  // Apply permutation to all meta types
  permute_type_array<4, sizeof...(VertexMetaTypes) + 4, VertexMetaTypes...>(
      old_soa, new_soa, permutation, blocks_count);
  */

#if RELABEL_VERBOSE
  printf("\nRelabeling: \n");
  for (int i = 0; i < nV(); i++) {
    printf("%d|", i);
  };
  printf("\n");
  thrust::device_ptr<int> p_ptr(relabeling);
  thrust::copy(p_ptr, p_ptr + nV(), std::ostream_iterator<int>(std::cout, "|"));

  printf("\nOld Vertex Degrees: \n");
  thrust::device_ptr<degree_t> old_vertex_degrees(old_soa.template get<0>());
  thrust::copy(old_vertex_degrees, old_vertex_degrees + nV(),
               std::ostream_iterator<degree_t>(std::cout, "|"));

  printf("\nNew Vertex Degrees: \n");
  thrust::device_ptr<degree_t> new_vertex_degrees(new_soa.template get<0>());
  thrust::copy(new_vertex_degrees, new_vertex_degrees + nV(),
               std::ostream_iterator<degree_t>(std::cout, "|"));
  printf("\n");
#endif

  // New the vertex data is permuted
  _vertex_data.template copy(new_vertex_data);

  // We need to change all adj-lists to the new vertex ids
  _ba_manager.relabel(relabeling);

  // Sort all blocks in id-order
  if (sort_edges)
    _ba_manager.sort();
}

} // namespace gpu

} // namespace hornet

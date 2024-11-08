/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include "BinarySearchKernel.cuh"
#include "StandardAPI.hpp"
#include <Device/Primitives/CubWrapper.cuh> //xlib::CubExclusiveSum
#include <Device/Util/DeviceProperties.cuh> //xlib::SMemPerBlock
#include <thrust/scan.h>                    //thrust::exclusive_scan

namespace hornets_nest {
namespace load_balancing {

template <typename HornetClass>
BinarySearch::BinarySearch(HornetClass &hornet,
                           const float work_factor) noexcept {
  // static_assert(IsHornet<HornetClass>::value,
  //              "BinarySearch: parameter is not an instance of Hornet Class");
  d_work.resize(work_factor * hornet.nV());
}

inline BinarySearch::~BinarySearch() noexcept {
  // hornets_nest::gpu::free(_d_work);
}

template <typename HornetClass, typename Operator, typename vid_t>
void BinarySearch::apply(HornetClass &hornet, const vid_t *d_input,
                         int num_vertices, const Operator &op) const noexcept {

  d_work.resize(num_vertices + 1);
  assert(num_vertices < _work_size && "BinarySearch (work queue) too small");
  // printf("[BSLD] Resized d_work and prefixsum to %d\n", num_vertices + 1);

  if (d_input != nullptr) {
    kernel::computeWorkKernel<<<xlib::ceil_div<BLOCK_SIZE>(num_vertices),
                                BLOCK_SIZE>>>(
        hornet.device(), d_input, num_vertices, d_work.data().get());
  } else {
    kernel::computeWorkKernel<<<xlib::ceil_div<BLOCK_SIZE>(num_vertices),
                                BLOCK_SIZE>>>(hornet.device(), num_vertices,
                                              d_work.data().get());
  }
  CHECK_CUDA_ERROR

#if 0
    thrust::host_vector<vid_t> h_input(num_vertices);
    gpu::copyToHost(d_input, num_vertices, h_input.data().get());

    printf("[BSLD] Executed computeWorkKernel, result vector:\n");
    thrust::host_vector<int> h_work = d_work;
    for (unsigned int i = 0; i < h_work.size() - 1; i++)
        printf("[%6d]\tvertex: %6d | work_size: %4d\n", i, h_input[i], h_work[i]);
#endif

  thrust::exclusive_scan(d_work.begin(), d_work.end(), d_work.begin());
  CHECK_CUDA_ERROR

#if 0
    printf("[BSLD] Executed PrefixSum, result vector:\n");
    h_work = d_work;
    for (unsigned int i = 0; i < h_work.size() - 1; i++)
        printf("[%6d]\tvertex: %6d | work_size: %4d\n", i, h_input[i], h_work[i]);
    printf("[BSLD] Total work: %d\n", h_work[h_work.size() - 1]);
#endif

  int total_work;
  cuMemcpyToHost(d_work.data().get() + num_vertices, total_work);
  if (total_work == 0)
    return;

  const unsigned int BLOCK_COUNT = xlib::ceil_div(total_work, BLOCK_SIZE);
  if (d_input != nullptr) {
    kernel::binarySearchKernel<BLOCK_SIZE><<<BLOCK_COUNT, BLOCK_SIZE>>>(
        hornet.device(), d_input, d_work.data().get(), num_vertices + 1, op);
  } else {
    kernel::binarySearchKernel<BLOCK_SIZE><<<BLOCK_COUNT, BLOCK_SIZE>>>(
        hornet.device(), d_work.data().get(), num_vertices + 1, op);
  }
  CHECK_CUDA_ERROR
}

template <typename HornetClass, typename Operator>
void BinarySearch::apply(HornetClass &hornet,
                         const Operator &op) const noexcept {
  apply<HornetClass, Operator, int>(hornet, nullptr, (int)hornet.nV(), op);
}

template <typename HornetClass, typename Operator,
          template <typename> typename Update, typename vid_t>
void BinarySearch::apply(HornetClass &hornet, Update<vid_t> &batch,
                         const Operator &op, bool reverse) {

  auto soa_ptr = batch.in_edge().get_soa_ptr();
  vid_t *d_input;

  if (reverse)
    d_input = soa_ptr.template get<1>();
  else
    d_input = soa_ptr.template get<0>();

  apply<HornetClass, Operator, vid_t>(hornet, d_input, batch.size(), op);
}

} // namespace load_balancing
} // namespace hornets_nest

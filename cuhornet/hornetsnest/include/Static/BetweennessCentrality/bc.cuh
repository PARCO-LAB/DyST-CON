/**
 * @brief
 * @author Oded Green                                                       <br>
 *   NVIDIA Corporation                                                     <br>
 *   ogreen@nvidia.com
 *   @author Muhammad Osama Sakhi                                           <br>
 *   Georgia Institute of Technology                                        <br>
 * @date July, 2018
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
 *
 * @file
 */
#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>

namespace hornets_nest {

using vid_t = int;
using HornetGraph = ::hornet::gpu::Hornet<vid_t>;
using HornetInit = ::hornet::HornetInit<vid_t>;

using paths_t = unsigned long long int;
using bc_t = double;
struct BCData {
  vid_t *d;
  vid_t *depth_indices;
  paths_t *sigma;
  bc_t *delta;
  bc_t *bc;
  vid_t root;
  degree_t currLevel;
  TwoLevelQueue<vid_t> queue;
};

class BCCentrality : public StaticAlgorithm<HornetGraph> {
  BufferPool pool;

public:
  BCCentrality(HornetGraph &hornet);

  ~BCCentrality();

  void setRoot(vid_t root_);

  void reset() override;
  void run() override;
  void release() override;
  bool validate() override;

  BCData bc_data() { return hd_BCData; };

  bc_t *getBCScores();
  paths_t *getSigmas();
  bc_t *getDeltas();

  vid_t getBestVertex();

private:
  load_balancing::BinarySearch load_balancing;

  HostDeviceVar<BCData> hd_BCData;
};

} // namespace hornets_nest

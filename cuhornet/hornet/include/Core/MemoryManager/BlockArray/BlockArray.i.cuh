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
namespace hornet {

#define BLOCK_ARRAY BlockArray<TypeList<Ts...>, device_t, degree_t>
#define B_A_MANAGER BlockArrayManager<TypeList<Ts...>, device_t, degree_t>

#define BLOCK_ARRAY_SORT_FIX 1

template <typename... Ts, DeviceType device_t, typename degree_t>
BLOCK_ARRAY::BlockArray(const int block_items,
                        const int blockarray_items) noexcept
    : _edge_data(blockarray_items), _bit_tree(block_items, blockarray_items) {

  CSoAPtr<Ts...> soa = _edge_data.get_soa_ptr();
  using T0 = typename xlib::SelectType<0, Ts...>::type;
  constexpr T0 limit = std::numeric_limits<T0>::max();

#if BLOCK_ARRAY_SORT_FIX
  if (device_t == DeviceType::DEVICE) {

    thrust::device_ptr<T0> ptr =
        thrust::device_pointer_cast(soa.template get<0>());
    thrust::fill_n(thrust::device, ptr, blockarray_items, limit);

    // cudaMemset(soa.template get<0>(), sizeof(T0) * blockarray_items, 0xFF);
  } else {

    T0 *ptr = soa.template get<0>();
    std::fill_n(ptr, blockarray_items, limit);
    // memset(soa.template get<0>(), sizeof(T0) * blockarray_items, 0xFF);
  }
#endif
}

template <typename... Ts, DeviceType device_t, typename degree_t>
BLOCK_ARRAY::BlockArray(const BLOCK_ARRAY &other) noexcept
    : _edge_data(other._edge_data), _bit_tree(other._bit_tree) {}

template <typename... Ts, DeviceType device_t, typename degree_t>
BLOCK_ARRAY::BlockArray(BLOCK_ARRAY &&other) noexcept
    : _edge_data(std::move(other._edge_data)),
      _bit_tree(std::move(other._bit_tree)) {}

template <typename... Ts, DeviceType device_t, typename degree_t>
xlib::byte_t *BLOCK_ARRAY::get_blockarray_ptr(void) noexcept {
  return reinterpret_cast<xlib::byte_t *>(
      _edge_data.get_soa_ptr().template get<0>());
}

template <typename... Ts, DeviceType device_t, typename degree_t>
int BLOCK_ARRAY::insert(void) noexcept {
  return _bit_tree.insert() << _bit_tree.get_log_block_items();
}

template <typename... Ts, DeviceType device_t, typename degree_t>
void BLOCK_ARRAY::remove(int offset) noexcept {
  _bit_tree.remove(offset);
}

template <typename... Ts, DeviceType device_t, typename degree_t>
int BLOCK_ARRAY::capacity(void) noexcept {
  return _edge_data.get_num_items();
}

template <typename... Ts, DeviceType device_t, typename degree_t>
size_t BLOCK_ARRAY::mem_size(void) noexcept {
  return xlib::SizeSum<Ts...>::value * capacity();
}

template <typename... Ts, DeviceType device_t, typename degree_t>
bool BLOCK_ARRAY::full(void) noexcept {
  return _bit_tree.full();
}

template <typename... Ts, DeviceType device_t, typename degree_t>
CSoAData<TypeList<Ts...>, device_t> &BLOCK_ARRAY::get_soa_data(void) noexcept {
  return _edge_data;
}

#define BLOCK_ARRAY_SORT_OLD 0
#define BLOCK_ARRAY_SORT_VERBOSE 0
#define BLOCK_ARRAY_SORT_SKIP_SIZE 64

template <typename... Ts, DeviceType device_t, typename degree_t>
void BLOCK_ARRAY::sort(void) {
#if BLOCK_ARRAY_SORT_OLD
  _edge_data.segmented_sort(_bit_tree.get_log_block_items());
#else

  static_assert(sizeof...(Ts) == 1, "Can sort only edges without metatypes");
  using T0 = typename xlib::SelectType<0, Ts...>::type;

  int size = _edge_data.get_num_items();
  int segment_size = 1 << _bit_tree.get_log_block_items();
  int segments_count = size / segment_size;

  // Skip sorting blocks with not too many nodes
  if (segment_size < BLOCK_ARRAY_SORT_SKIP_SIZE)
    return;

#if BLOCK_ARRAY_SORT_VERBOSE
  printf("Block Array Sort(segment_size: %d, size: %d, segments_count: "
         "%d)\n",
         segment_size, size, segments_count);
#endif

  // Create temporary copy of edges
  CSoAData<TypeList<Ts...>, DeviceType::DEVICE> tmp_edge_data(size);
  CSoAPtr<Ts...> tmp_soa = tmp_edge_data.get_soa_ptr();
  CSoAPtr<Ts...> soa = _edge_data.get_soa_ptr();
  tmp_edge_data.copy(soa, device_t, size);

  T0 *keys_in = tmp_soa.template get<0>();
  T0 *keys_out = soa.template get<0>();

  // Calculate offset of all segments
  cudaStream_t stream{nullptr};
  rmm::device_vector<degree_t> segments(segments_count + 1);
  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(segments_count + 1),
                    thrust::make_constant_iterator(segment_size),
                    segments.begin(), thrust::multiplies<degree_t>{});

#if BLOCK_ARRAY_SORT_VERBOSE
  printf("Block Array Relabel Sort SegmentsBeg[..10]:\n\t");
  thrust::copy(segments.begin(), segments.begin() + 10,
               std::ostream_iterator<int>(std::cout, ", "));
  printf("\n");
#endif

  // TODO: Solve bug if nodes are sorted in ascending order,
  // the zero-initialized values after the node's adjlist
  // are sorted as well, resulting in a bunch of zeros at
  // the beginning of the sorted adjlist.
  // For now the problem has been "fixed" by initializing
  // the edge_data SoA arrays with sentinel values bigger
  // than any node id.

  // Calculate required temporary buffer memory
  size_t temp_buffer_size = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(
      NULL, temp_buffer_size, keys_in, keys_out, size, segments.size() - 1,
      segments.data().get(), segments.data().get() + 1);

  // Sort nodes
  rmm::device_buffer temp_buffer(temp_buffer_size, rmm::cuda_stream_view{});
  cub::DeviceSegmentedRadixSort::SortKeys(
      temp_buffer.data(), temp_buffer_size, keys_in, keys_out, size,
      segments.size() - 1, segments.data().get(), segments.data().get() + 1);

#endif
}

#define BLOCK_ARRAY_RELABEL_VERBOSE 1
#define BLOCK_ARRAY_RELABEL_BLOCK_SIZE 1024
#define BLOCK_ARRAY_RELABEL_BLOCK_WORK 20
#define BLOCK_ARRAY_RELABEL_WORK                                               \
  (BLOCK_ARRAY_RELABEL_BLOCK_SIZE * BLOCK_ARRAY_RELABEL_BLOCK_WORK)

template <typename T, T invalid_id>
static __global__ void kernel_relabel(const T *relabeling, T *vertices,
                                      const int N) {

  int grid_size = blockDim.x * gridDim.x;
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = global_idx; i < N; i += grid_size) {
    // printf("Relabeling at index %d: %d -> %d\n", i, vertices[i],
    //        relabeling[vertices[i]]);
    if (vertices[i] != invalid_id) {
      vertices[i] = relabeling[vertices[i]];
    }
  }
}

template <typename... Ts, DeviceType device_t, typename degree_t>
void BLOCK_ARRAY::relabel(int *relabeling) {
  using T0 = typename xlib::SelectType<0, Ts...>::type;
  constexpr T0 limit = std::numeric_limits<T0>::max();

  CSoAPtr<Ts...> soa = _edge_data.get_soa_ptr();
  const int size = _edge_data.get_num_items();

  const int block_count =
      (size + BLOCK_ARRAY_RELABEL_WORK - 1) / BLOCK_ARRAY_RELABEL_WORK;
  kernel_relabel<T0, limit><<<block_count, BLOCK_ARRAY_RELABEL_BLOCK_SIZE>>>(
      relabeling, soa.template get<0>(), size);
}

#define BLOCK_ARRAY_FILL_BLOCK_SIZE 1024
#define BLOCK_ARRAY_FILL_BLOCK_WORK 10
#define BLOCK_ARRAY_FILL_WORK                                                  \
  (BLOCK_ARRAY_FILL_BLOCK_SIZE * BLOCK_ARRAY_FILL_BLOCK_WORK)

template <typename T>
__global__ void kernel_fill(T *array, const int N, T value) {

  int grid_size = blockDim.x * gridDim.x;
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = global_idx; i < N; i += grid_size) {
    array[i] = value;
  }
}

template <typename... Ts, DeviceType device_t, typename degree_t>
void BLOCK_ARRAY::fill_max() {
  using T0 = typename xlib::SelectType<0, Ts...>::type;
  CSoAPtr<Ts...> soa = _edge_data.get_soa_ptr();

  printf("FILLING ARRAY(size: %d)...", 1 << _bit_tree.get_log_block_items());

  const int size = _edge_data.get_num_items();
  const int blocks_count =
      (size + BLOCK_ARRAY_FILL_WORK - 1) / BLOCK_ARRAY_FILL_WORK;

  T0 *vids = soa.template get<0>();
  kernel_fill<T0><<<blocks_count, BLOCK_ARRAY_FILL_BLOCK_SIZE>>>(
      vids, size, std::numeric_limits<T0>::max());

  printf("DONE\n");
}

//==============================================================================

template <typename degree_t>
int find_bin(const degree_t requested_degree) noexcept {
  return (requested_degree <= MIN_EDGES_PER_BLOCK
              ? 0
              : xlib::ceil_log2(requested_degree) -
                    xlib::Log2<MIN_EDGES_PER_BLOCK>::value);
}

template <typename... Ts, DeviceType device_t, typename degree_t>
B_A_MANAGER::BlockArrayManager(const degree_t MaxEdgesPerBlockArray) noexcept
    : _MaxEdgesPerBlockArray(1 << xlib::ceil_log2(MaxEdgesPerBlockArray)),
      _largest_eb_size(1 << xlib::ceil_log2(MaxEdgesPerBlockArray)) {}

template <typename... Ts, DeviceType device_t, typename degree_t>
template <DeviceType d_t>
B_A_MANAGER::BlockArrayManager(
    const BlockArrayManager<TypeList<Ts...>, d_t, degree_t> &other) noexcept
    : _ba_map(other._ba_map) {}

template <typename... Ts, DeviceType device_t, typename degree_t>
template <DeviceType d_t>
B_A_MANAGER::BlockArrayManager(
    BlockArrayManager<TypeList<Ts...>, d_t, degree_t> &&other) noexcept
    : _ba_map(std::move(other._ba_map)) {}

template <typename... Ts, DeviceType device_t, typename degree_t>
EdgeAccessData<degree_t>
B_A_MANAGER::insert(const degree_t requested_degree) noexcept {
  if (requested_degree == 0) {
    EdgeAccessData<degree_t> ea = {0, 0, 0};
    return ea;
  }
  int bin_index = find_bin(requested_degree);
  for (auto &ba : _ba_map[bin_index]) {
    if (!ba.second.full()) {
      degree_t offset = ba.second.insert();
      EdgeAccessData<degree_t> ea = {ba.second.get_blockarray_ptr(), offset,
                                     ba.second.capacity()};
      return ea;
    }
  }
  _largest_eb_size =
      std::max(1 << xlib::ceil_log2(requested_degree), _largest_eb_size);
  BLOCK_ARRAY new_block_array(
      1 << xlib::ceil_log2(requested_degree),
      std::max(1 << xlib::ceil_log2(requested_degree), _MaxEdgesPerBlockArray));
  degree_t offset = new_block_array.insert();
  EdgeAccessData<degree_t> ea = {new_block_array.get_blockarray_ptr(), offset,
                                 new_block_array.capacity()};

  auto block_ptr = new_block_array.get_blockarray_ptr();
  _ba_map[bin_index].insert(
      std::make_pair(block_ptr, std::move(new_block_array)));

  // Fill with max val to solve sorting bug
  // new_block_array.fill_max();

  return ea;
}

template <typename... Ts, DeviceType device_t, typename degree_t>
void B_A_MANAGER::remove(degree_t degree, xlib::byte_t *edge_block_ptr,
                         degree_t vertex_offset) noexcept {
  int bin_index = find_bin(degree);
  auto &ba = _ba_map[bin_index].at(edge_block_ptr);
  ba.remove(vertex_offset);
}

template <typename... Ts, DeviceType device_t, typename degree_t>
degree_t B_A_MANAGER::largest_edge_block_size(void) noexcept {
  return _largest_eb_size;
}

template <typename... Ts, DeviceType device_t, typename degree_t>
void B_A_MANAGER::removeAll(void) noexcept {
  for (auto &b : _ba_map) {
    b.clear();
  }
}

template <typename... Ts, DeviceType device_t, typename degree_t>
void B_A_MANAGER::sort(void) {
  for (unsigned i = 0; i < _ba_map.size(); ++i) {
    for (auto &b : _ba_map[i]) {
      b.second.sort();
    }
  }
}

template <typename... Ts, DeviceType device_t, typename degree_t>
void B_A_MANAGER::relabel(int *permutation) {
  for (unsigned i = 0; i < _ba_map.size(); ++i) {
    for (auto &b : _ba_map[i]) {
      b.second.relabel(permutation);
    }
  }
}

#undef BLOCK_ARRAY
} // namespace hornet

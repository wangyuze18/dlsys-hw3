#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <limits> 

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if(gid < size){
    size_t ssize = shape.size;
    size_t pos = offset;
    size_t t = gid;
    for(int32_t i=ssize-1; i>=0; i--){
      if(i){ 
        size_t mod = t % shape.data[i];
        t /= shape.data[i];
        pos += mod * strides.data[i];
      }
      else{
        pos += t * strides.data[i];
      }
    }
    out[gid] = a[pos];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size){
      size_t ssize = shape.size;
      size_t pos = offset;
      size_t t = gid;
      for(int32_t i=ssize-1; i>=0; i--){
        if(i){ 
          size_t mod = t % shape.data[i];
          t /= shape.data[i];
          pos += mod * strides.data[i];
        }
        else{
          pos += t * strides.data[i];
        }
      }
      out[pos] = a[gid];
    }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                          VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out,  CudaVec shape,
                              CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size){
      size_t ssize = shape.size;
      size_t pos = offset;
      size_t t = gid;
      for(int32_t i=ssize-1; i>=0; i--){
        if(i){ 
          size_t mod = t % shape.data[i];
          t /= shape.data[i];
          pos += mod * strides.data[i];
        }
        else{
          pos += t * strides.data[i];
        }
      }
      out[pos] = val;
    }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape),
                          VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */
template <typename Op>
__global__ void EwiseOpKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid], b[gid]);
}

template <typename Op>
__global__ void EwiseFuncKernel(const scalar_t* a, scalar_t* out, size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid]);
}

template <typename Op>
__global__ void ScalarOpKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid], val);
}

#define DEFINE_EWISE_OP_FUNC(func_name, OpType) \
void func_name(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, OpType()); \
}

#define DEFINE_EWISE_FUNC_FUNC(func_name, OpType) \
void func_name(const CudaArray& a, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  EwiseFuncKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, OpType()); \
}

#define DEFINE_SCALAR_OP_FUNC(func_name, OpType) \
void func_name(const CudaArray& a, scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, OpType()); \
}

struct MulOp {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a * b; }
};

struct DivOp {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a / b; }
};

struct MaximumOp {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return max(a, b); }
  __device__ scalar_t identity() { return -INFINITY;}
};

struct EqOp {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a == b; }
};

struct GeOp {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a >= b; }
};

struct PowerOp {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return pow(a, b); }
};

struct LogOp {
  __device__ scalar_t operator()(scalar_t a) const { return log(a); }
};

struct ExpOp {
  __device__ scalar_t operator()(scalar_t a) const { return exp(a); }
};

struct TanhOp {
  __device__ scalar_t operator()(scalar_t a) const { return tanh(a); }
};

struct AddOp {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a + b; }
  __device__ scalar_t identity() { return 0;}
};
DEFINE_EWISE_OP_FUNC(EwiseMul, MulOp)
DEFINE_EWISE_OP_FUNC(EwiseDiv, DivOp)
DEFINE_EWISE_OP_FUNC(EwiseMaximum, MaximumOp)
DEFINE_EWISE_OP_FUNC(EwiseEq, EqOp)
DEFINE_EWISE_OP_FUNC(EwiseGe, GeOp)

DEFINE_SCALAR_OP_FUNC(ScalarMul, MulOp)
DEFINE_SCALAR_OP_FUNC(ScalarDiv, DivOp)
DEFINE_SCALAR_OP_FUNC(ScalarMaximum, MaximumOp)
DEFINE_SCALAR_OP_FUNC(ScalarEq, EqOp)
DEFINE_SCALAR_OP_FUNC(ScalarGe, GeOp)
DEFINE_SCALAR_OP_FUNC(ScalarPower, PowerOp)

DEFINE_EWISE_FUNC_FUNC(EwiseLog, LogOp)
DEFINE_EWISE_FUNC_FUNC(EwiseExp, ExpOp)
DEFINE_EWISE_FUNC_FUNC(EwiseTanh, TanhOp)

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
#define L 64
#define S 8
__global__ void MatmulKernel(const scalar_t* A, const scalar_t* B, scalar_t* out, size_t M, size_t N, size_t P){
  // block level
  __shared__ scalar_t sA[S][L], sB[S][L]; 
  size_t x_block = blockIdx.x;
  size_t y_block = blockIdx.y;
  size_t nthreads = blockDim.y * blockDim.x;
  size_t tid = threadIdx.x + blockDim.x * threadIdx.y;
  scalar_t c[TILE][TILE]={0};
  scalar_t a[TILE],b[TILE];

  for(size_t ko=0; ko<N; ko+=S){
    for(size_t i=0; i< S * L / nthreads; i++){
      int y = (i * nthreads + tid) / L;
      int x = (i * nthreads + tid) % L;
      sA[y][x] = (y_block * L + x < M && ko + y < N) ? 
                    A[(y_block * L + x) * N + (ko + y)] : 0;
      sB[y][x] = (ko + y < N && x_block * L + x < P) ? 
                  B[(ko + y) * P + (x_block * L + x)] : 0;
    }
    __syncthreads();
    for(size_t ki=0; ki<S; ki++){
      for(int i=0; i<TILE; i++){
        a[i] = sA[ki][threadIdx.y * TILE + i];
        b[i] = sB[ki][threadIdx.x * TILE + i];
      }
      for(size_t y=0; y<TILE; y++){
        for(size_t x=0; x<TILE; x++){
          c[y][x] += a[y] * b[x];
        }
      }
    }
    __syncthreads();
  }

  size_t y_base = y_block * L + threadIdx.y * TILE; 
  size_t x_base = x_block * L + threadIdx.x * TILE;
  for(size_t i=0; i<TILE; i++){
    for(size_t j=0; j<TILE; j++){
      if(y_base + i < M && x_base + j < P)
        out[(y_base + i) * P + x_base + j] = c[i][j];
    }
  }

}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
    CudaDims dims;
    dims.block = dim3(L / TILE, L / TILE);
    dims.grid = dim3((P + L - 1) / L
              ,(M + L - 1) / L);
    MatmulKernel<<<dims.grid, dims.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
template <typename Op>
__device__ void WarpReduce(volatile scalar_t* sdata,size_t tid, Op op){
    sdata[tid] = op(sdata[tid], sdata[tid+32]);
    sdata[tid] = op(sdata[tid], sdata[tid+16]);
    sdata[tid] = op(sdata[tid], sdata[tid+8]);
    sdata[tid] = op(sdata[tid], sdata[tid+4]);
    sdata[tid] = op(sdata[tid], sdata[tid+2]);
    sdata[tid] = op(sdata[tid], sdata[tid+1]);
}

template <typename Op>
__global__ void ReduceKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, Op op){
  size_t tid = threadIdx.x;
  size_t colid = blockIdx.y * blockDim.x + threadIdx.x;
  size_t gid = blockIdx.x * gridDim.y * reduce_size + blockIdx.y + threadIdx.x;
  extern __shared__ scalar_t sdata[];
  sdata[tid] = colid < reduce_size ? a[gid] : op.identity();
  __syncthreads();
  for(size_t s=blockDim.x/2; s>32; s>>=1){
      if(tid < s){
        sdata[tid] = op(sdata[tid], sdata[tid+s]);
      }
      __syncthreads();
  }
  if(tid<32) WarpReduce(sdata, tid, op);
  if(tid==0) out[blockIdx.x] = sdata[0];
}


void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION​​
    size_t current_size = reduce_size;
    scalar_t *d_input = a.ptr, *d_output = out->ptr;
    if(current_size < BASE_THREAD_NUM){
      dim3 block(BASE_THREAD_NUM);
      size_t num_blocks = (current_size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
      dim3 grid(out->size, num_blocks);

      ReduceKernel<<<grid, block, BASE_THREAD_NUM * ELEM_SIZE>>>(
          d_input, d_output, 
          current_size, MaximumOp()
      );
    }

    size_t temp_size = (current_size + BASE_THREAD_NUM) / BASE_THREAD_NUM;
    scalar_t *d_temp = nullptr;
    cudaMalloc(&d_temp, temp_size * ELEM_SIZE);

    while (current_size >= BASE_THREAD_NUM) {
        dim3 block(BASE_THREAD_NUM, 1, 1);  
        size_t num_blocks = (current_size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
        dim3 grid(out->size, num_blocks, 1); 

        ReduceKernel<<<grid, block, BASE_THREAD_NUM * ELEM_SIZE>>>(
            d_input, (current_size > BASE_THREAD_NUM) ? d_temp : d_output, 
            current_size, MaximumOp()
        );

        d_input = d_temp;
        current_size = num_blocks;
    }

    cudaFree(d_temp);

  /// END SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
    size_t current_size = reduce_size;
    scalar_t *d_input = a.ptr, *d_output = out->ptr;
    if(current_size < BASE_THREAD_NUM){
      dim3 block(BASE_THREAD_NUM);
      size_t num_blocks = (current_size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
      dim3 grid(out->size, num_blocks);

      ReduceKernel<<<grid, block, BASE_THREAD_NUM * ELEM_SIZE>>>(
          d_input, d_output, 
          current_size, AddOp()
      );
    }

    size_t temp_size = (current_size + BASE_THREAD_NUM) / BASE_THREAD_NUM;
    scalar_t *d_temp = nullptr;
    cudaMalloc(&d_temp, temp_size * ELEM_SIZE);

    while (current_size >= BASE_THREAD_NUM) {
        dim3 block(BASE_THREAD_NUM, 1, 1);  
        size_t num_blocks = (current_size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
        dim3 grid(out->size, num_blocks, 1); 

        ReduceKernel<<<grid, block, BASE_THREAD_NUM * ELEM_SIZE>>>(
            d_input, (current_size > BASE_THREAD_NUM) ? d_temp : d_output, 
            current_size, AddOp()
        );

        d_input = d_temp;
        current_size = num_blocks;
    }

    cudaFree(d_temp);

  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}

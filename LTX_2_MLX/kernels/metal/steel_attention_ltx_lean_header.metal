// Lean BF16-only MLX STEEL subset for the LTX-2.3 no-mask attention path.
// Derived from Apple MLX STEEL attention sources.
// Copyright (c) 2024-25 Apple Inc.
// SPDX-License-Identifier: MIT
#define MLX_METAL_JIT 1
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

#pragma METAL internals : enable

namespace mlx {
namespace steel {

template <
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / tgp_size,
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BF16BlockLoaderT {
  STEEL_CONST short vec_size = n_reads;

  const int src_ld;
  const int tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup bfloat* dst;
  const device bfloat* src;

  METAL_FUNC BF16BlockLoaderT(
      const device bfloat* src_,
      const int src_ld_,
      threadgroup bfloat* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(src_ + bi * src_ld + bj) {}

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = bfloat(0);
        }
      }
      return;
    }

    bool tmp_idx[vec_size];
    bfloat tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_idx[j] ? tmp_val[j] : bfloat(0);
      }
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
  }
};

typedef metal::vec<float, 2> mma_frag_t;
typedef metal::simdgroup_matrix<float, 8, 8> mma_mat_t;

METAL_FUNC static constexpr short2 mma_coord(
    ushort simd_lane_id [[thread_index_in_simdgroup]]) {
  const short qid = simd_lane_id / 4;
  const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
  const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
  return short2{fn, fm};
}

METAL_FUNC static constexpr void
load_fragment(thread mma_frag_t& dst, const threadgroup bfloat* src) {
  dst[0] = static_cast<float>(src[0]);
  dst[1] = static_cast<float>(src[1]);
}

METAL_FUNC static constexpr void
store_fragment(const thread mma_frag_t& src, device bfloat* dst) {
  dst[0] = static_cast<bfloat>(src[0]);
  dst[1] = static_cast<bfloat>(src[1]);
}

METAL_FUNC static constexpr void store_fragment_safe(
    const thread mma_frag_t& src,
    device bfloat* dst,
    const int lim_y,
    const int off_y) {
  if (off_y < lim_y) {
    dst[off_y] = static_cast<bfloat>(src[0]);
  }
  if ((off_y + 1) < lim_y) {
    dst[off_y + 1] = static_cast<bfloat>(src[1]);
  }
}

METAL_FUNC static void mma_fragment(
    thread mma_frag_t& D,
    thread mma_frag_t& A,
    thread mma_frag_t& B,
    thread mma_frag_t& C) {
  mma_mat_t D_mat;
  mma_mat_t A_mat;
  mma_mat_t B_mat;
  mma_mat_t C_mat;

  reinterpret_cast<thread mma_frag_t&>(A_mat.thread_elements()) = A;
  reinterpret_cast<thread mma_frag_t&>(B_mat.thread_elements()) = B;
  reinterpret_cast<thread mma_frag_t&>(C_mat.thread_elements()) = C;

  simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
  D = reinterpret_cast<thread mma_frag_t&>(D_mat.thread_elements());
}

template <typename Op>
METAL_FUNC static constexpr void row_reduce_fragment(
    thread const mma_frag_t& inp_vals,
    thread float* reduced_vals) {
  float thr_reduce = Op::apply(inp_vals.x, inp_vals.y);
  float qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
  qgr_reduce = Op::apply(thr_reduce, qgr_reduce);
  float sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
  sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);
  reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
}

template <typename Op>
METAL_FUNC static constexpr void
row_bin_op_fragment(thread mma_frag_t& inp_vals, thread float* row_vals) {
  inp_vals[0] = Op::apply(inp_vals[0], row_vals[0]);
  inp_vals[1] = Op::apply(inp_vals[1], row_vals[0]);
}

template <int COLS>
struct RowTile {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;
  STEEL_CONST int kElemCols = 2;
  STEEL_CONST int kTileCols = COLS;
  STEEL_CONST int kElemsPerTile = COLS * kElemCols;
  STEEL_CONST int kRowsPerThread = 1;

  mma_frag_t val_frags[COLS];

  METAL_FUNC RowTile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j] = mma_frag_t(0);
    }
  }

  METAL_FUNC constexpr thread mma_frag_t& frag_at(const short j) {
    return val_frags[j];
  }

  METAL_FUNC thread float* elems() {
    return reinterpret_cast<thread float*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread float vals[kRowsPerThread]) const {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      row_reduce_fragment<Op>(val_frags[j], vals);
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread float vals[kRowsPerThread]) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      row_bin_op_fragment<Op>(val_frags[j], vals);
    }
  }

  METAL_FUNC void load(const threadgroup bfloat* src) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      load_fragment(frag_at(j), &src[j * kFragCols]);
    }
  }

  METAL_FUNC void store(device bfloat* dst) const {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      store_fragment(val_frags[j], &dst[j * kFragCols]);
    }
  }

  METAL_FUNC void
  store_safe(device bfloat* dst, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < COLS; ++j) {
      store_fragment_safe(
          val_frags[j],
          dst,
          dst_tile_dims.x,
          j * kFragCols);
    }
  }
};

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable

using namespace mlx::steel;

struct MaxOp {
  METAL_FUNC static constexpr float apply(float x, float y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  METAL_FUNC static constexpr float apply(float x, float y) {
    return x + y;
  }
};

struct MulOp {
  METAL_FUNC static constexpr float apply(float x, float y) {
    return x * y;
  }
};

struct ExpSubOp {
  METAL_FUNC static constexpr float apply(float x, float y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  METAL_FUNC static constexpr float apply(float x, float y) {
    return x / y;
  }
};

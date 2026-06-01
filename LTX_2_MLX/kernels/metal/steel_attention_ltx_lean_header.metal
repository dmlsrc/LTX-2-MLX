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
    short kRows,
    short kCols,
    short kDstStrRow,
    short kDstStrCol,
    bool kReductionDim>
struct BF16BlockLoader {
  STEEL_CONST short kThreads = 256;
  STEEL_CONST short vec_size = (kRows * kCols) / kThreads;
  STEEL_CONST short kThreadCols = kCols / vec_size;
  STEEL_CONST short kThreadRows = kThreads / kThreadCols;
  static_assert(kThreadRows == kRows, "Lean loader expects one row stripe.");

  const int src_ld;
  const int tile_stride;
  const short thread_idx;
  const short row;
  const short col;
  threadgroup bfloat* dst;
  const device bfloat* src;

  METAL_FUNC BF16BlockLoader(
      const device bfloat* src_,
      const int src_ld_,
      threadgroup bfloat* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(kReductionDim ? kCols : kRows * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        row(thread_idx / kThreadCols),
        col(vec_size * (thread_idx % kThreadCols)),
        dst(dst_ + row * kDstStrRow + col * kDstStrCol),
        src(src_ + row * src_ld + col) {}

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++) {
      dst[j * kDstStrCol] = src[j];
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(col, row);
    const bool row_valid = src_tile_dim.y > 0;

    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++) {
      dst[j * kDstStrCol] =
          (row_valid && j < src_tile_dim.x) ? src[j] : bfloat(0);
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

template <int COLS>
struct RowTile {
  STEEL_CONST int kFragCols = 8;

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

  METAL_FUNC void row_max(thread float* vals) const {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      float thr_reduce = metal::max(val_frags[j].x, val_frags[j].y);
      float qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = metal::max(thr_reduce, qgr_reduce);
      float sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = metal::max(qgr_reduce, sgr_reduce);
      vals[0] = metal::max(vals[0], sgr_reduce);
    }
  }

  METAL_FUNC void row_sum(thread float* vals) const {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      float thr_reduce = val_frags[j].x + val_frags[j].y;
      float qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = thr_reduce + qgr_reduce;
      float sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = qgr_reduce + sgr_reduce;
      vals[0] += sgr_reduce;
    }
  }

  METAL_FUNC void scale_by(float scale) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] *= scale;
      val_frags[j][1] *= scale;
    }
  }

  METAL_FUNC void exp2_sub(thread float* vals) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] = fast::exp2(val_frags[j][0] - vals[0]);
      val_frags[j][1] = fast::exp2(val_frags[j][1] - vals[0]);
    }
  }

  METAL_FUNC void mul_by(thread float* vals) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] *= vals[0];
      val_frags[j][1] *= vals[0];
    }
  }

  METAL_FUNC void div_by(thread float* vals) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] /= vals[0];
      val_frags[j][1] /= vals[0];
    }
  }

  METAL_FUNC void load(const threadgroup bfloat* src) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      const short off = j * kFragCols;
      val_frags[j][0] = static_cast<float>(src[off]);
      val_frags[j][1] = static_cast<float>(src[off + 1]);
    }
  }

  METAL_FUNC void store(device bfloat* dst) const {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      const short off = j * kFragCols;
      dst[off] = static_cast<bfloat>(val_frags[j][0]);
      dst[off + 1] = static_cast<bfloat>(val_frags[j][1]);
    }
  }

  METAL_FUNC void
  store_safe(device bfloat* dst, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < COLS; ++j) {
      const short off = j * kFragCols;
      if (off < dst_tile_dims.x) {
        dst[off] = static_cast<bfloat>(val_frags[j][0]);
      }
      if ((off + 1) < dst_tile_dims.x) {
        dst[off + 1] = static_cast<bfloat>(val_frags[j][1]);
      }
    }
  }
};

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable

using namespace mlx::steel;

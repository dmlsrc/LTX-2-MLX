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

template <typename T, int kFragRows, int kFragCols>
struct BaseMMAFrag;

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;
  STEEL_CONST int kElemsPerFrag = 2;
  STEEL_CONST int kElemRows = 1;
  STEEL_CONST int kElemCols = 2;

  typedef metal::simdgroup_matrix<T, 8, 8> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;

  METAL_FUNC static constexpr short2 get_coord(
      ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <int str_x, int str_y>
  METAL_FUNC static constexpr void
  load(thread frag_type& dst, const threadgroup bfloat* src) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = static_cast<T>(src[i * str_x + j * str_y]);
      }
    }
  }

  METAL_FUNC static constexpr void
  store(const thread frag_type& src, device bfloat* dst, const int ld) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * ld + j] = static_cast<bfloat>(src[i * kElemCols + j]);
      }
    }
  }

  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src,
      device bfloat* dst,
      const int ld,
      const int lim_x,
      const int lim_y,
      const int off_x,
      const int off_y) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * ld + off_y + j] =
              static_cast<bfloat>(src[i * kElemCols + j]);
        }
      }
    }
  }

  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread frag_type& A,
      thread frag_type& B,
      thread frag_type& C) {
    mat_type D_mat;
    mat_type A_mat;
    mat_type B_mat;
    mat_type C_mat;

    reinterpret_cast<thread frag_type&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread frag_type&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(C_mat.thread_elements()) = C;

    simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_reduce(
      thread const frag_type& inp_vals,
      thread T* reduced_vals) {
    T thr_reduce = Op::apply(inp_vals.x, inp_vals.y);
    T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = Op::apply(thr_reduce, qgr_reduce);
    T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);
    reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_bin_op(
      thread frag_type& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

template <
    typename T,
    int kTileRows_,
    int kTileCols_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  STEEL_CONST int kFragRows = MMAFrag_t::kFragRows;
  STEEL_CONST int kFragCols = MMAFrag_t::kFragCols;
  STEEL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;
  STEEL_CONST int kTileRows = kTileRows_;
  STEEL_CONST int kTileCols = kTileCols_;
  STEEL_CONST int kNumFrags = kTileRows * kTileCols;
  STEEL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;
  STEEL_CONST int kRowsPerThread = kTileRows * MMAFrag_t::kElemRows;

  typedef typename MMAFrag_t::frag_type frag_type;
  frag_type val_frags[kNumFrags];

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread T vals[kRowsPerThread]) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kRowsPerThread]) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup bfloat* src) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template load<str_x, str_y>(
            frag_at(i, j),
            &(src[(i * kFragRows) * w_x * str_x + (j * kFragCols) * w_y * str_y]));
      }
    }
  }

  template <int w_x, int w_y>
  METAL_FUNC void store(device bfloat* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld);
      }
    }
  }

  template <int w_x, int w_y>
  METAL_FUNC void
  store_safe(device bfloat* dst, const int ld, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
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

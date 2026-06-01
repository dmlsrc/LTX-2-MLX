// BF16-only body fragment consumed by mx.fast.metal_kernel.
// MLX supplies the kernel wrapper and input/output pointer declarations.
// Derived from Apple MLX STEEL attention sources.
// Copyright (c) 2024-25 Apple Inc.
// SPDX-License-Identifier: MIT
  const int H = Q_shape[1];
  const int qL = Q_shape[2];
  const int kL = K_shape[2];

  const int Q_stride_h = Q_strides[1];
  const int Q_stride_t = Q_strides[2];
  const int K_stride_h = K_strides[1];
  const int K_stride_t = K_strides[2];
  const int V_stride_h = V_strides[1];
  const int V_stride_t = V_strides[2];
  const int O_stride_h = BD;
  const int O_stride_t = H * BD;

  constexpr int BQ = 64;
  constexpr int BK = 32;
  constexpr int WM = 8;

  const int NK = (kL + BK - 1) / BK;
  const int NQ_aligned = qL / BQ;
  const int NK_aligned = kL / BK;
  const int qL_rem = qL - NQ_aligned * BQ;
  const int kL_rem = kL - NK_aligned * BK;

  uint simd_lane_id = thread_index_in_simdgroup;
  uint simd_group_id = simdgroup_index_in_threadgroup;
  uint3 tid = threadgroup_position_in_grid;

  using AccumType = float;

  Q += tid.y * Q_stride_h + tid.x * BQ * Q_stride_t;
  K += tid.y * K_stride_h;
  V += tid.y * V_stride_h;
  O += tid.y * O_stride_h + tid.x * BQ * O_stride_t;

  constexpr short padQ = 16 / sizeof(bfloat);
  constexpr short padK = 16 / sizeof(bfloat);
  constexpr short padV = 16 / sizeof(bfloat);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * BD;
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup bfloat Q_smem[BQ * (BD + padQ)];
  threadgroup bfloat KV_smem[tgp_mem_s];

  threadgroup bfloat* Qs = Q_smem;
  threadgroup bfloat* Ks = KV_smem;
  threadgroup bfloat* Vs = KV_smem;

  using QBlockLoader = BF16BlockLoader<
      BQ,
      BD,
      LDQ_tgp,
      1,
      true>;

  using KBlockLoader = BF16BlockLoader<
      BK,
      BD,
      1,
      LDK_tgp,
      false>;

  using VBlockLoader = BF16BlockLoader<
      BK,
      BD,
      LDV_tgp,
      1,
      false>;

  QBlockLoader loader_q(Q, Q_stride_t, Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(K, K_stride_t, Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(V, V_stride_t, Vs, simd_group_id, simd_lane_id);

  const AccumType scale = (1.0f / sqrt(float(BD))) * M_LOG2E_F;

  constexpr short kFragSize = 8;

  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;
  static_assert(BQ == WM * kFragSize, "Lean LTX STEEL attention expects one Q row.");

  RowTile<1> Qtile;
  RowTile<TK> Ktile;
  RowTile<TK> Stile;
  RowTile<1> Vtile;
  RowTile<TD> Otile;
  Otile.clear();

  const short2 simd_coord = mma_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!AlignQ && int(tid.x) == NQ_aligned) {
    loader_q.load_safe(short2(BD, qL_rem));
  } else {
    loader_q.load_unsafe();
  }

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = -3.4028234663852886e+38F;
  }

  for (int kb = 0; kb < NK; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!AlignK && kb == NK_aligned) {
      loader_k.load_safe(short2(BD, kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.load(&Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.load(&Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        mma_fragment(
            Stile.frag_at(ik),
            Qtile.frag_at(0),
            Ktile.frag_at(ik),
            Stile.frag_at(ik));
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    if (!AlignK && kb == NK_aligned) {
      using stile_t = decltype(Stile);
      constexpr AccumType neg_inf = -3.4028234663852886e+38F;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < stile_t::kTileCols; j++) {
        short col_pos = sn + (j * stile_t::kFragCols);
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < stile_t::kElemCols; jj++) {
          if ((col_pos + jj) >= kL_rem) {
            Stile.frag_at(j)[jj] = neg_inf;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!AlignK && kb == NK_aligned) {
      loader_v.load_safe(short2(BD, kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    Otile.template row_bin_op<MulOp>(factor);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        if constexpr (BD == 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }

        const short kk = ik * kFragSize;
        const short dd = id * kFragSize;

        Vtile.load(&Vs[Vs_offset + kk * LDV_tgp + dd]);

        if constexpr (BD == 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }

        mma_fragment(
            Otile.frag_at(id),
            Stile.frag_at(ik),
            Vtile.frag_at(0),
            Otile.frag_at(id));
      }
    }

    loader_k.next();
    loader_v.next();
  }

  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  O += (tm + sm) * O_stride_t + sn;

  if (!AlignQ && int(tid.x) == NQ_aligned) {
    auto dst_tile_dims = short2(BD - sn, qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
      return;
    }

    Otile.store_safe(O, dst_tile_dims);
  } else {
    Otile.store(O);
  }

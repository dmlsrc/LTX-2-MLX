// BF16-only body fragment consumed by mx.fast.metal_kernel.
// MLX supplies the kernel wrapper and input/output pointer declarations.
// Derived from Apple MLX STEEL attention sources.
// Copyright (c) 2024-25 Apple Inc.
// SPDX-License-Identifier: MIT
  constexpr int H = 32;
  const int qL = Q_shape[2];
  const int kL = K_shape[2];

  constexpr int BQ = 64;
  constexpr int BK = 32;

  const int Q_stride_h = Q_strides[1];
  const int Q_stride_t = Q_strides[2];
  const int K_stride_h = K_strides[1];
  const int K_stride_t = K_strides[2];
  const int V_stride_h = V_strides[1];
  const int V_stride_t = V_strides[2];
  constexpr int O_stride_h = BD;
  constexpr int O_stride_t = H * BD;

  const int NQ_aligned = qL / BQ;
  const int NK_aligned = kL / BK;
  const int qL_rem = qL - NQ_aligned * BQ;
  const int kL_rem = kL - NK_aligned * BK;

  uint simd_lane_id = thread_index_in_simdgroup;
  uint simd_group_id = simdgroup_index_in_threadgroup;
  uint3 tid = threadgroup_position_in_grid;

  Q += tid.y * Q_stride_h + tid.x * BQ * Q_stride_t;
  K += tid.y * K_stride_h;
  V += tid.y * V_stride_h;
  O += tid.y * O_stride_h + tid.x * BQ * O_stride_t;

  constexpr short pad = 16 / sizeof(bfloat);

  constexpr short LDQ_tgp = BD + pad;
  constexpr short LDK_tgp = BK + pad;
  constexpr short LDV_tgp = BD + pad;

  threadgroup bfloat Q_smem[BQ * LDQ_tgp];
  threadgroup bfloat KV_smem[BD * LDK_tgp];

  using QBlockLoader = BF16BlockLoader<BQ, BD, LDQ_tgp, 1>;
  using KBlockLoader = BF16BlockLoader<BK, BD, 1, LDK_tgp>;
  using VBlockLoader = BF16BlockLoader<BK, BD, LDV_tgp, 1>;

  QBlockLoader loader_q(Q, Q_stride_t, Q_smem, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(K, K_stride_t, KV_smem, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(V, V_stride_t, KV_smem, simd_group_id, simd_lane_id);

  const float scale = (1.0f / sqrt(float(BD))) * M_LOG2E_F;

  constexpr short kFragSize = 8;

  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  RowTile<1> Qtile;
  RowTile<TK> Ktile;
  RowTile<TK> Stile;
  RowTile<1> Vtile;
  RowTile<TD> Otile;
  Otile.clear();

  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + ((simd_lane_id / 2) % 4);
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
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

  constexpr float neg_inf = -3.4028234663852886e+38F;

  float max_score[1] = {neg_inf};
  float sum_score[1] = {0};

  // Keep the hot K loop branch-free.  LTX stage grids are often only a few
  // tokens short of a BK multiple; handling that single partial tile in a
  // separate epilogue lets the full tiles compile like AlignK=true.
  //
  // This intentionally duplicates the tile body instead of hiding it behind a
  // macro/template.  The full-tile path below has no k-tail checks or masks in
  // the inner loop; the epilogue below handles the one partial BK tile exactly.
  for (int kb = 0; kb < NK_aligned; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_k.load_unsafe();

    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stile was just cleared, so the first QK slice can use multiply instead
    // of multiply-accumulate.  Keep this as a prologue; an in-loop dd==0
    // branch was slower on the stage-2 D128 shape.
    simdgroup_barrier(mem_flags::mem_none);

    Qtile.load(&Q_smem[Qs_offset]);
    Ktile.load(&KV_smem[Ks_offset]);

    simdgroup_barrier(mem_flags::mem_none);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      mma_fragment_mul(
          Stile.frag_at(ik),
          Qtile.frag_at(0),
          Ktile.frag_at(ik));
    }

    STEEL_PRAGMA_UNROLL
    for (short dd = 1; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.load(&Q_smem[Qs_offset + dd * Qs_tile_stride]);
      Ktile.load(&KV_smem[Ks_offset + dd * Ks_tile_stride]);

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

    Stile.scale_by(scale);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_v.load_unsafe();

    float new_max[1] = {max_score[0]};
    float factor[1];

    Stile.row_max(new_max);
    Stile.exp2_sub(new_max);

    factor[0] = fast::exp2(max_score[0] - new_max[0]);
    max_score[0] = new_max[0];

    float sum_score_tmp[1] = {0};
    Stile.row_sum(sum_score_tmp);

    sum_score[0] = sum_score[0] * factor[0] + sum_score_tmp[0];

    Otile.mul_by(factor);

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

        Vtile.load(&KV_smem[Vs_offset + kk * LDV_tgp + dd]);

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

  if constexpr (!AlignK) {
    // Partial K tile epilogue.  Same online-softmax update as the full-tile
    // loop, but K/V loads are bounds-checked and invalid score lanes are set
    // to -inf before the row max/sum update.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_k.load_safe(short2(BD, kL_rem));

    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_barrier(mem_flags::mem_none);

    Qtile.load(&Q_smem[Qs_offset]);
    Ktile.load(&KV_smem[Ks_offset]);

    simdgroup_barrier(mem_flags::mem_none);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      mma_fragment_mul(
          Stile.frag_at(ik),
          Qtile.frag_at(0),
          Ktile.frag_at(ik));
    }

    STEEL_PRAGMA_UNROLL
    for (short dd = 1; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.load(&Q_smem[Qs_offset + dd * Qs_tile_stride]);
      Ktile.load(&KV_smem[Ks_offset + dd * Ks_tile_stride]);

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

    Stile.scale_by(scale);

    // Mask score lanes beyond kL.  Only the final BK tile reaches this block.
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      short col_pos = sn + (j * kFragSize);
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < 2; jj++) {
        if ((col_pos + jj) >= kL_rem) {
          Stile.frag_at(j)[jj] = neg_inf;
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_v.load_safe(short2(BD, kL_rem));

    float new_max[1] = {max_score[0]};
    float factor[1];

    Stile.row_max(new_max);
    Stile.exp2_sub(new_max);

    factor[0] = fast::exp2(max_score[0] - new_max[0]);
    max_score[0] = new_max[0];

    float sum_score_tmp[1] = {0};
    Stile.row_sum(sum_score_tmp);

    sum_score[0] = sum_score[0] * factor[0] + sum_score_tmp[0];

    Otile.mul_by(factor);

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

        Vtile.load(&KV_smem[Vs_offset + kk * LDV_tgp + dd]);

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

  Otile.div_by(sum_score);
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

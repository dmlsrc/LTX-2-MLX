
// BF16 D128 body fragment consumed by mx.fast.metal_kernel.

  uint simd_lane_id = thread_index_in_simdgroup;
  uint simd_group_id = simdgroup_index_in_threadgroup;
  uint3 tid = threadgroup_position_in_grid;

  constexpr int BD = 128;
  constexpr int BQ = 80;
  constexpr int BK = 40;
  constexpr int WM = 10;
  constexpr int Q_PAD = 8;
  constexpr int K_PAD = 2;
  constexpr int V_PAD = 8;
  constexpr int Q_ACTIVE_THREADS = 320;
  constexpr int K_ACTIVE_THREADS = 320;
  constexpr int V_ACTIVE_THREADS = 320;
  constexpr bool AllActiveLoaders = true;
  constexpr int Q_FULL_TILES_CONST = -1;
  constexpr int K_FULL_TILES_CONST = -1;
  constexpr int Q_REM_CONST = -1;
  constexpr int K_REM_CONST = -1;
  constexpr bool SkipUnitFactor = false;

  constexpr int H = 32;
  static_assert(BD == 128, "This kernel currently supports D=128.");
  static_assert(BK % 8 == 0, "BK must match the 8x8 fragment tiling.");
  static_assert(BQ == WM * 8, "This kernel maps one 8-row Q stripe per simdgroup.");
  static_assert(Q_ACTIVE_THREADS == WM * 32, "Q loading uses every thread.");
  static_assert(
      !AllActiveLoaders ||
          (Q_ACTIVE_THREADS == WM * 32 && K_ACTIVE_THREADS == WM * 32 &&
           V_ACTIVE_THREADS == WM * 32),
      "All-active loader variant requires every thread to participate.");
  static_assert(
      (Q_FULL_TILES_CONST >= 0) == (Q_REM_CONST >= 0),
      "Exact Q specialization needs both full-tile count and remainder.");
  static_assert(
      (K_FULL_TILES_CONST >= 0) == (K_REM_CONST >= 0),
      "Exact K specialization needs both full-tile count and remainder.");

  const int qL = Q_shape[2];
  const int kL = K_shape[2];
  const int Q_stride_h = Q_strides[1];
  const int Q_stride_t = Q_strides[2];
  const int K_stride_h = K_strides[1];
  const int K_stride_t = K_strides[2];
  const int V_stride_h = V_strides[1];
  const int V_stride_t = V_strides[2];
  constexpr int O_stride_h = BD;
  constexpr int O_stride_t = H * BD;

  constexpr bool exact_q = Q_FULL_TILES_CONST >= 0;
  constexpr bool exact_k = K_FULL_TILES_CONST >= 0;
  const int NQ_aligned = exact_q ? Q_FULL_TILES_CONST : qL / BQ;
  const int NK_aligned = exact_k ? K_FULL_TILES_CONST : kL / BK;
  const int qL_rem = exact_q ? Q_REM_CONST : qL - NQ_aligned * BQ;
  const int kL_rem = exact_k ? K_REM_CONST : kL - NK_aligned * BK;

  Q += tid.y * Q_stride_h + tid.x * BQ * Q_stride_t;
  K += tid.y * K_stride_h;
  V += tid.y * V_stride_h;
  O += tid.y * O_stride_h + tid.x * BQ * O_stride_t;

  constexpr short LDQ_tgp = BD + Q_PAD;
  constexpr short LDK_tgp = BK + K_PAD;
  constexpr short LDV_tgp = BD + V_PAD;
  constexpr int K_smem_elems = BD * LDK_tgp;
  constexpr int V_smem_elems = BK * LDV_tgp;
  constexpr int KV_smem_elems =
      K_smem_elems > V_smem_elems ? K_smem_elems : V_smem_elems;
  static_assert(
      (BQ * LDQ_tgp + KV_smem_elems) * int(sizeof(bfloat)) <= 32768,
      "Threadgroup memory budget exceeded.");

  threadgroup bfloat Q_smem[BQ * LDQ_tgp];
  threadgroup bfloat KV_smem[KV_smem_elems];

  using QBlockLoader =
      BF16BlockLoader<BQ, BD, LDQ_tgp, 1, Q_ACTIVE_THREADS, AllActiveLoaders>;
  using KBlockLoader =
      BF16BlockLoader<BK, BD, 1, LDK_tgp, K_ACTIVE_THREADS, AllActiveLoaders>;
  using VBlockLoader =
      BF16BlockLoader<BK, BD, LDV_tgp, 1, V_ACTIVE_THREADS, AllActiveLoaders>;

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

  const short qid = short(simd_lane_id / 4);
  const short sm = (qid & 4) + ((simd_lane_id / 2) % 4);
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
  const short tm = kFragSize * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if constexpr (!AlignQ) {
    if (int(tid.x) == NQ_aligned) {
      loader_q.load_safe(short2(BD, qL_rem));
    } else {
      loader_q.load_unsafe();
    }
  } else {
    loader_q.load_unsafe();
  }

  constexpr float neg_inf = -3.4028234663852886e+38F;
  float max_score[1] = {neg_inf};
  float sum_score[1] = {0};

  for (int kb = 0; kb < NK_aligned; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_k.load_unsafe();

    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
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

    if constexpr (SkipUnitFactor) {
      factor[0] = (new_max[0] == max_score[0])
          ? 1.0f
          : fast::exp2(max_score[0] - new_max[0]);
    } else {
      factor[0] = fast::exp2(max_score[0] - new_max[0]);
    }
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
        simdgroup_barrier(mem_flags::mem_none);

        const short kk = ik * kFragSize;
        const short dd = id * kFragSize;

        Vtile.load(&KV_smem[Vs_offset + kk * LDV_tgp + dd]);

        simdgroup_barrier(mem_flags::mem_none);

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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_k.load_safe(short2(BD, kL_rem));

    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
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

    if constexpr (SkipUnitFactor) {
      factor[0] = (new_max[0] == max_score[0])
          ? 1.0f
          : fast::exp2(max_score[0] - new_max[0]);
    } else {
      factor[0] = fast::exp2(max_score[0] - new_max[0]);
    }
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
        simdgroup_barrier(mem_flags::mem_none);

        const short kk = ik * kFragSize;
        const short dd = id * kFragSize;

        Vtile.load(&KV_smem[Vs_offset + kk * LDV_tgp + dd]);

        simdgroup_barrier(mem_flags::mem_none);

        mma_fragment(
            Otile.frag_at(id),
            Stile.frag_at(ik),
            Vtile.frag_at(0),
            Otile.frag_at(id));
      }
    }
  }

  Otile.div_by(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  O += (tm + sm) * O_stride_t + sn;

  if constexpr (!AlignQ) {
    if (int(tid.x) == NQ_aligned) {
      auto dst_tile_dims = short2(BD - sn, qL_rem - (tm + sm));

      if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
        return;
      }

      Otile.store_safe(O, dst_tile_dims);
    } else {
      Otile.store(O);
    }
  } else {
    Otile.store(O);
  }

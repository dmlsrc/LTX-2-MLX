// Minimal os_signpost wrapper for ctypes use.
//
// os_signpost is a macro-based API that takes the calling image's
// __dso_handle as an implicit parameter, so it cannot be called
// directly from Python ctypes.  This shim is built into a dylib so
// the macros expand with this image's own __dso_handle.
//
// All phase names are baked into per-function symbols so each phase
// gets a stable static name in the Instruments timeline.  This is
// the os_signpost contract: name must be a string literal at compile
// time.  Adding a new phase = add a new pair of functions here.
//
// Build:
//   clang -O2 -shared -fPIC -o _signpost.dylib _signpost.c

#include <os/log.h>
#include <os/signpost.h>
#include <stdint.h>

#define LTX_PROFILE_SUBSYSTEM "ltx"

static os_log_t _ltx_poi_log = NULL;

static inline void _ltx_init(void) {
    if (!_ltx_poi_log) {
        _ltx_poi_log = os_log_create(LTX_PROFILE_SUBSYSTEM,
                                     OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    }
}

uint64_t ltx_signpost_id_generate(void) {
    _ltx_init();
    return os_signpost_id_generate(_ltx_poi_log);
}

int ltx_signpost_enabled(void) {
    _ltx_init();
    return os_signpost_enabled(_ltx_poi_log) ? 1 : 0;
}

// Phase-named interval pairs.  Each pair shares a stable static name
// so Instruments groups all calls under one phase row.

#define LTX_PHASE(NAME) \
    void ltx_signpost_begin_##NAME(uint64_t sid) { \
        _ltx_init(); \
        os_signpost_interval_begin(_ltx_poi_log, sid, #NAME); \
    } \
    void ltx_signpost_end_##NAME(uint64_t sid) { \
        _ltx_init(); \
        os_signpost_interval_end(_ltx_poi_log, sid, #NAME); \
    }

LTX_PHASE(video_self_attn)
LTX_PHASE(video_text_ca)
LTX_PHASE(audio_self_attn)
LTX_PHASE(audio_text_ca)
LTX_PHASE(a2v_cross)
LTX_PHASE(v2a_cross)
LTX_PHASE(video_ff)
LTX_PHASE(audio_ff)

// Generic event for top-level step / block markers.
void ltx_signpost_event_step_begin(uint64_t sid, uint64_t step_idx) {
    _ltx_init();
    os_signpost_event_emit(_ltx_poi_log, sid, "step_begin",
                           "step=%llu", (unsigned long long)step_idx);
}
void ltx_signpost_event_step_end(uint64_t sid, uint64_t step_idx) {
    _ltx_init();
    os_signpost_event_emit(_ltx_poi_log, sid, "step_end",
                           "step=%llu", (unsigned long long)step_idx);
}
void ltx_signpost_event_block(uint64_t sid, uint64_t block_idx) {
    _ltx_init();
    os_signpost_event_emit(_ltx_poi_log, sid, "block",
                           "block=%llu", (unsigned long long)block_idx);
}

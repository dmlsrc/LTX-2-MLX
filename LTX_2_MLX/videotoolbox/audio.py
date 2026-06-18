"""In-memory PCM audio wrapped as CMSampleBuffers for AVAssetWriter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from ._compat import CoreAudio, CoreMedia, require_pyobjc

# CoreAudio FormatID constants (avoid importing the whole module just for these)
AUDIO_FORMAT_LPCM = 1819304813     # 'lpcm' kAudioFormatLinearPCM
AUDIO_FORMAT_AAC = 1633772320      # 'aac ' kAudioFormatMPEG4AAC
AUDIO_FORMAT_ALAC = 1634492771     # 'alac' kAudioFormatAppleLossless


class AudioTrack:
    """In-memory audio decoded from a latent. No disk WAV unless save_wav()
    is called explicitly.

    Constructed from a (channels, samples) float32 array (mlx or numpy). Builds
    CMSampleBuffers on demand via `make_sample_buffer(start_frame, end_frame)`
    - the AVWriter's GCD audio pump pulls these in chunks as the encoder
    drains.

    Format: interleaved 32-bit float PCM in the source sample rate. The
    writer's audio output settings (ALAC / AAC) handle the encode-time
    conversion.
    """

    def __init__(self, waveform: Any, sample_rate: int):
        require_pyobjc()
        # Accept an mlx or numpy (channels, samples) array; normalize to mlx f32.
        w = mx.array(waveform, dtype=mx.float32)
        if w.ndim != 2:
            raise ValueError(
                f"AudioTrack expects (channels, samples); got {w.shape}"
            )
        self.sample_rate = int(sample_rate)
        self.channels = int(w.shape[0])
        self.n_samples = int(w.shape[1])
        # Interleave: (channels, samples) -> (samples, channels) row-major bytes,
        # straight from the MLX buffer.
        self._bytes = bytes(memoryview(mx.contiguous(mx.transpose(w))))
        bytes_per_frame = 4 * self.channels

        asbd = CoreAudio.AudioStreamBasicDescription(
            float(self.sample_rate),
            AUDIO_FORMAT_LPCM,
            CoreAudio.kAudioFormatFlagIsFloat | CoreAudio.kAudioFormatFlagIsPacked,
            bytes_per_frame,   # mBytesPerPacket
            1,                 # mFramesPerPacket
            bytes_per_frame,   # mBytesPerFrame
            self.channels,
            32,                # mBitsPerChannel
            0,
        )
        err, fmt = CoreMedia.CMAudioFormatDescriptionCreate(
            None, asbd, 0, None, 0, None, None, None,
        )
        if err != 0 or fmt is None:
            raise RuntimeError(f"CMAudioFormatDescriptionCreate failed: status={err}")
        self.format_desc = fmt

    def save_wav(self, path: Path) -> None:
        """Write the in-memory PCM out as a float32 WAV (for --save-audio-sidecar)."""
        from LTX_2_MLX.video_encoder import write_wav_float32

        samples = mx.array(memoryview(self._bytes).cast("f")).reshape(
            self.n_samples, self.channels,
        )
        write_wav_float32(mx.transpose(samples), path, self.sample_rate)

    def make_sample_buffer(self, start_frame: int, end_frame: int) -> Any:
        """Build a CMSampleBuffer for audio frames [start_frame, end_frame).

        Returns None if the range is empty. Caller is responsible for
        appendSampleBuffer-ing it to an AVAssetWriterInput.
        """
        n = end_frame - start_frame
        if n <= 0:
            return None
        bytes_per_frame = 4 * self.channels
        chunk_bytes = self._bytes[start_frame * bytes_per_frame: end_frame * bytes_per_frame]
        data_len = len(chunk_bytes)

        err, block = CoreMedia.CMBlockBufferCreateWithMemoryBlock(
            None, None, data_len, None, None, 0, data_len, 1, None,
        )
        if err != 0 or block is None:
            raise RuntimeError(f"CMBlockBufferCreateWithMemoryBlock failed: {err}")
        err = CoreMedia.CMBlockBufferReplaceDataBytes(chunk_bytes, block, 0, data_len)
        if err != 0:
            raise RuntimeError(f"CMBlockBufferReplaceDataBytes failed: {err}")

        pts = CoreMedia.CMTimeMake(start_frame, self.sample_rate)
        err, sample_buf = CoreMedia.CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            None, block, self.format_desc, n, pts, None, None,
        )
        if err != 0 or sample_buf is None:
            raise RuntimeError(
                f"CMAudioSampleBufferCreateReadyWithPacketDescriptions failed: {err}"
            )
        return sample_buf


def audio_writer_settings(codec: str, sample_rate: int, channels: int) -> dict:
    """AVAssetWriterInput output settings for the configured audio codec."""
    require_pyobjc()
    from ._compat import av  # late import so the module loads without pyobjc

    if codec == "alac":
        return {
            av.AVFormatIDKey: AUDIO_FORMAT_ALAC,
            av.AVSampleRateKey: float(sample_rate),
            av.AVNumberOfChannelsKey: channels,
            av.AVEncoderBitDepthHintKey: 24,
        }
    if codec == "aac":
        return {
            av.AVFormatIDKey: AUDIO_FORMAT_AAC,
            av.AVSampleRateKey: float(sample_rate),
            av.AVNumberOfChannelsKey: channels,
            av.AVEncoderBitRateKey: 256000,
        }
    raise ValueError(f"Unknown audio codec {codec!r}")

"""Generation pipelines for LTX-2 MLX."""

from .av_pipeline import (
    AVCFGConfig,
    AVPipeline,
    create_av_pipeline,
)
from .av_pipeline import (
    ImageCondition as AVImageCondition,
)
from .ic_lora import (
    ICLoraConfig,
    ICLoraPipeline,
    ImageCondition,
    VideoCondition,
    create_ic_lora_pipeline,
    create_image_conditionings,
    create_video_conditionings,
    load_video_tensor,
)
from .keyframe_interpolation import (
    Keyframe,
    KeyframeInterpolationConfig,
    KeyframeInterpolationPipeline,
    create_keyframe_conditionings,
    create_keyframe_pipeline,
    load_image_as_tensor,
)
from .two_stage import (
    ImageCondition as TwoStageImageCondition,
)
from .two_stage import (
    TwoStageCFGConfig,
    TwoStagePipeline,
    create_two_stage_pipeline,
)

__all__ = [
    # Keyframe interpolation
    "KeyframeInterpolationConfig",
    "KeyframeInterpolationPipeline",
    "Keyframe",
    "create_keyframe_pipeline",
    "load_image_as_tensor",
    "create_keyframe_conditionings",
    # IC-LoRA
    "ICLoraConfig",
    "ICLoraPipeline",
    "ImageCondition",
    "VideoCondition",
    "create_ic_lora_pipeline",
    "load_video_tensor",
    "create_image_conditionings",
    "create_video_conditionings",
    # AV pipeline (CFG single-pass, also routes distilled + text-to-video)
    "AVCFGConfig",
    "AVPipeline",
    "AVImageCondition",
    "create_av_pipeline",
    # Two-stage CFG
    "TwoStageCFGConfig",
    "TwoStagePipeline",
    "TwoStageImageCondition",
    "create_two_stage_pipeline",
    # HQ Two-Stage (Res2s)
    "TI2VidHQConfig",
    "TI2VidHQPipeline",
    # Retake
    "RetakeConfig",
    "RetakePipeline",
    "TemporalRegionMask",
    # Audio-to-Video
    "A2VidConfig",
    "A2VidPipelineTwoStage",
]

from .a2vid_two_stage import A2VidConfig, A2VidPipelineTwoStage
from .retake import RetakeConfig, RetakePipeline, TemporalRegionMask
from .ti2vid_hq import TI2VidHQConfig, TI2VidHQPipeline

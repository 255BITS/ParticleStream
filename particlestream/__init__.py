"""ParticleStream: single-stream (batch_size=1) sequential image and memory-GAN benchmark."""
from .models import FastWeightMemory, StreamImageCritic, StreamImageGenerator
from .online import OnlineSession
from .memory import CausalStateMemory, make_stream_memory
from .transport import TransportGANTrainer
from .feature_prior import FeatureConditionalReconstruction
from .challenges import (
    CIRCLE_WAYPOINTS,
    CircleOrbitStream,
    circle_orbit_coords,
    circle_orbit_templates,
    extract_circle_coords,
    render_circle_orbit_gif,
    score_circle_orbit,
)
from .streams import (
    ForkStream,
    SequentialBarsStream,
    TranslatingBarStream,
    bars4_templates,
    fork_templates,
    score_fork_rollout,
    score_multimodal_bars,
    score_translating_next,
    translating_bar_cycle,
)

__all__ = [
    "CIRCLE_WAYPOINTS",
    "CircleOrbitStream",
    "FastWeightMemory",
    "OnlineSession",
    "CausalStateMemory",
    "make_stream_memory",
    "TransportGANTrainer",
    "FeatureConditionalReconstruction",
    "ForkStream",
    "SequentialBarsStream",
    "StreamImageCritic",
    "StreamImageGenerator",
    "TranslatingBarStream",
    "bars4_templates",
    "circle_orbit_coords",
    "circle_orbit_templates",
    "extract_circle_coords",
    "fork_templates",
    "render_circle_orbit_gif",
    "score_circle_orbit",
    "score_fork_rollout",
    "score_multimodal_bars",
    "score_translating_next",
    "translating_bar_cycle",
]

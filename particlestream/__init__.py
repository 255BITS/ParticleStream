"""ParticleStream: single-stream (batch_size=1) sequential image and memory-GAN benchmark."""
from .models import FastWeightMemory, StreamImageCritic, StreamImageGenerator
from .online import OnlineSession
from .memory import CausalStateMemory, make_stream_memory
from .transport import TransportGANTrainer
from .feature_prior import FeatureConditionalReconstruction
from .gail_prior import GAILConditionalReconstruction
from .residual_prior import ResidualFeatureConditionalReconstruction
from .subspace_prior import SubspaceResidualConditionalReconstruction
from .challenges import (
    CIRCLE_WAYPOINTS,
    SQUARE_WAYPOINTS,
    CircleOrbitStream,
    SquareCircleCycleStream,
    circle_orbit_coords,
    circle_orbit_templates,
    extract_circle_coords,
    render_circle_orbit_gif,
    render_square_circle_cycle_gif,
    score_circle_orbit,
    score_square_circle_cycle,
    square_orbit_coords,
    square_orbit_templates,
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
    "SQUARE_WAYPOINTS",
    "CircleOrbitStream",
    "SquareCircleCycleStream",
    "FastWeightMemory",
    "OnlineSession",
    "CausalStateMemory",
    "make_stream_memory",
    "TransportGANTrainer",
    "FeatureConditionalReconstruction",
    "GAILConditionalReconstruction",
    "ResidualFeatureConditionalReconstruction",
    "SubspaceResidualConditionalReconstruction",
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
    "render_square_circle_cycle_gif",
    "score_circle_orbit",
    "score_fork_rollout",
    "score_multimodal_bars",
    "score_square_circle_cycle",
    "score_translating_next",
    "square_orbit_coords",
    "square_orbit_templates",
    "translating_bar_cycle",
]

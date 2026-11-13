"""Standalone visual MuJoCo robotics demo for ParticleStream (batch_size=1, replay=0).

Trains two strictly online ParticleGAN learners simultaneously from scratch on a
single chronological FetchPickAndPlace-v4 stream (480 transitions, batch_size=1,
zero replay buffer, outside particlegan_forge):

1. **3D Holographic Particle Dynamics (`RoboticsSession` + `GAIL-BCAP`)**:
   32 particles predict action-conditioned 3D gripper & block displacements
   before each physics step, plus 6-step closed-loop trajectory rollouts with
   zero teacher forcing, projected directly into the 3D MuJoCo camera view.
2. **Live RGB Pixel Dreamer (`TransportGANTrainer` + `RGBGAILReconstruction`)**:
   A sample-independent CoordConv + FiLM residual generator dreams the upcoming
   48x64 RGB camera frame *before* the simulator advances, crystallizing from
   random neural static at Step 1 into the moving robot arm and lifted block.

Usage:
    python demos/mujoco_pick_and_place_demo.py --device cuda
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from particlegan_forge import NamedStreams  # noqa: E402
from particlestream import robotics_tasks as rdata  # noqa: E402
from particlestream.forge import StreamWorkspace  # noqa: E402
from particlestream.gail_prior import GAILConditionalReconstruction  # noqa: E402
from particlestream.minimal_adapter import _build, _probe  # noqa: E402
from particlestream.minimal_online import isolated_session, prefix_context  # noqa: E402
from particlestream.robotics_adapter import RoboticsSession  # noqa: E402
from particlestream.tier1 import REGISTRY, _init_module  # noqa: E402
from particlestream.transport import TransportGANTrainer  # noqa: E402


DREAM_H, DREAM_W = 48, 64
HERO_W, HERO_H = 552, 414
CANVAS_W, CANVAS_H = 1080, 560
ROLLOUT_HORIZON = 6
SHOWCASE_EPISODES = (0, 2, 6, 11)  # Episodes 1, 3, 7, 12 (0-indexed)


# ---------------------------------------------------------------------------
# Live RGB Pixel Dreamer (strictly batch_size=1, zero replay)
# ---------------------------------------------------------------------------


def pool_rgb_to_8x8_descriptor(frame: torch.Tensor) -> torch.Tensor:
    """Deterministically summarize a (1, 3, 48, 64) RGB frame into (1, 1, 8, 8)."""
    p4 = F.adaptive_avg_pool2d(frame, (4, 4)).reshape(1, 48)
    p2_max = F.adaptive_max_pool2d(frame, (2, 2)).reshape(1, 12)
    ch_mean = frame.mean(dim=(2, 3))
    ch_std = frame.std(dim=(2, 3), unbiased=False).narrow(1, 0, 1)
    return torch.cat([p4, p2_max, ch_mean, ch_std], dim=1).reshape(1, 1, 8, 8)


class RGBGAILReconstruction(GAILConditionalReconstruction):
    """GAIL-BCAP online conditional prior operating on single (1, 3, 48, 64) RGB frames."""

    @contextmanager
    def training_target(self, frame: torch.Tensor):
        if self._target is not None:
            raise RuntimeError("a GAIL-prior update is already active")
        if (
            not isinstance(frame, torch.Tensor)
            or tuple(frame.shape) != (1, 3, DREAM_H, DREAM_W)
            or frame.requires_grad
            or not bool(torch.isfinite(frame).all())
        ):
            raise ValueError(f"RGB transport requires one detached finite (1,3,{DREAM_H},{DREAM_W}) frame")
        self._pending_reward = self._pending_style = self._pending_gain = None
        self._pending_basis_row = self._pending_basis_gain = None
        self._target = frame.detach().clone()
        self._phase, self._selected = 0, None
        try:
            self._pending_context = self._current_context()
            if self.subspace_dims > 0:
                cand = self._candidate_innovation(frame)
                if cand is not None:
                    self._pending_basis_row, self._pending_basis_gain = cand
            self._pending_feature = self.encode(frame)[0]
            yield
            if self._phase != 2 or self._selected is None:
                raise RuntimeError("transport requires one D draw and one G draw")
            with torch.no_grad():
                self.assignment_mass.mul_(self.usage_decay)
                self.assignment_mass[self._selected].add_(1)
                self.assignment_updates.add_(1)
            self._commit_observation(self._pending_context, self._pending_feature)
        finally:
            self._target, self._selected, self._phase = None, None, 0
            self._pending_context = self._pending_feature = None
            self._pending_reward = self._pending_style = self._pending_gain = None
            self._pending_basis_row = self._pending_basis_gain = None

    @torch.no_grad()
    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        desc = pool_rgb_to_8x8_descriptor(frame) if tuple(frame.shape) == (1, 3, DREAM_H, DREAM_W) else frame
        return super().encode(desc)

    @torch.no_grad()
    def _candidate_innovation(self, frame: torch.Tensor):
        desc = pool_rgb_to_8x8_descriptor(frame) if tuple(frame.shape) == (1, 3, DREAM_H, DREAM_W) else frame
        return super()._candidate_innovation(desc)


class RGBStreamGenerator(nn.Module):
    """Sample-independent 3x48x64 CoordConv + FiLM residual generator (no BatchNorm, batch_size=1)."""

    def __init__(self, z_dim: int = 24, width: int = 48, context_dim: int = 29):
        super().__init__()
        self.z_dim = int(z_dim)
        self.width = int(width)
        self.context_dim = int(context_dim)
        self.fc = nn.Linear(self.z_dim + self.context_dim, self.width * 2 * 6 * 8)
        self.proj = nn.Conv2d(self.width * 2, self.width, 1)
        self.c1 = nn.Conv2d(self.width + 2, self.width, 3, padding=1)
        self.film1 = nn.Linear(self.context_dim, self.width * 2)
        self.c2 = nn.Conv2d(self.width + 2, self.width, 3, padding=1)
        self.film2 = nn.Linear(self.context_dim, self.width * 2)
        self.c3 = nn.Conv2d(self.width + 2, self.width, 3, padding=1)
        self.film3 = nn.Linear(self.context_dim, self.width * 2)
        self.c4 = nn.Conv2d(self.width + 2, self.width // 2, 3, padding=1)
        self.head = nn.Conv2d(self.width // 2, 3, 3, padding=1)
        self.register_buffer("active_context", torch.zeros(1, max(1, self.context_dim)))

    def set_context(self, context: torch.Tensor) -> None:
        if self.context_dim > 0:
            self.active_context.copy_(context.detach()[:1, : self.context_dim])

    @staticmethod
    def _coord_grid(batch: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(batch, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(batch, 1, h, w)
        return torch.cat([xs, ys], dim=1)

    @staticmethod
    def _apply_film(feat: torch.Tensor, film_layer: nn.Linear, ctx: torch.Tensor) -> torch.Tensor:
        gamma, beta = film_layer(ctx).chunk(2, dim=-1)
        return feat * (1.0 + 0.25 * torch.tanh(gamma[:, :, None, None])) + 0.25 * beta[:, :, None, None]

    def forward(self, z: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if len(z) != 1:
            raise ValueError("RGBStreamGenerator strictly enforces batch_size=1")
        ctx = (context if context is not None else self.active_context.expand(len(z), -1)).detach()
        inp = torch.cat([z, ctx], dim=-1)
        h = F.leaky_relu(self.fc(inp), 0.2).reshape(-1, self.width * 2, 6, 8)
        h = self.proj(h)
        h = h + F.leaky_relu(self._apply_film(self.c1(torch.cat([h, self._coord_grid(len(z), 6, 8, z.device, z.dtype)], 1)), self.film1, ctx), 0.2)
        h = F.interpolate(h, scale_factor=2, mode="nearest")  # 12x16
        h = h + F.leaky_relu(self._apply_film(self.c2(torch.cat([h, self._coord_grid(len(z), 12, 16, z.device, z.dtype)], 1)), self.film2, ctx), 0.2)
        h = F.interpolate(h, scale_factor=2, mode="nearest")  # 24x32
        h = h + F.leaky_relu(self._apply_film(self.c3(torch.cat([h, self._coord_grid(len(z), 24, 32, z.device, z.dtype)], 1)), self.film3, ctx), 0.2)
        h = F.interpolate(h, scale_factor=2, mode="nearest")  # 48x64
        h = F.leaky_relu(self.c4(torch.cat([h, self._coord_grid(len(z), 48, 64, z.device, z.dtype)], 1)), 0.2)
        return torch.sigmoid(self.head(h))


class RGBStreamCritic(nn.Module):
    """Sample-independent 3x48x64 convolutional critic (no BatchNorm, batch_size=1)."""

    def __init__(self, width: int = 24, context_dim: int = 29):
        super().__init__()
        self.context_dim = int(context_dim)
        self.c1 = nn.Conv2d(3, width, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1)
        self.c4 = nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(width * 2 * 12 + self.context_dim, width * 2)
        self.fc2 = nn.Linear(width * 2, 1)
        self.register_buffer("active_context", torch.zeros(1, max(1, self.context_dim)))

    def set_context(self, context: torch.Tensor) -> None:
        if self.context_dim > 0:
            self.active_context.copy_(context.detach()[:1, : self.context_dim])

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if len(x) != 1:
            raise ValueError("RGBStreamCritic strictly enforces batch_size=1")
        h = F.leaky_relu(self.c1(x), 0.2)
        h = F.leaky_relu(self.c2(h), 0.2)
        h = F.leaky_relu(self.c3(h), 0.2)
        h = F.leaky_relu(self.c4(h), 0.2).flatten(1)
        ctx = (context if context is not None else self.active_context.expand(len(x), -1)).detach()
        return self.fc2(F.leaky_relu(self.fc1(torch.cat([h, ctx], dim=-1)), 0.2))


def build_rgb_context(mem_ctx: torch.Tensor, obs_dict: dict, action: np.ndarray, device: str) -> torch.Tensor:
    """Pre-step causal conditioning vector combining FastWeightMemory + normalized pre-step pose/action."""
    s = obs_dict["observation"]
    anchor = np.array([1.30, 0.75, 0.60], dtype=np.float32)
    grip = (s[:3] - anchor) * 3.0
    obj = (s[3:6] - anchor) * 3.0
    goal = (obs_dict["desired_goal"] - anchor) * 3.0
    pose = torch.tensor(np.concatenate([grip, obj, goal, action]), dtype=torch.float32, device=device).unsqueeze(0)
    return torch.cat([mem_ctx, pose], dim=1).detach()


# ---------------------------------------------------------------------------
# MuJoCo Studio Styling & Exact 3D -> 2D Camera Projection
# ---------------------------------------------------------------------------


def style_mujoco_scene(env) -> None:
    """Apply dark slate studio palette (#0b0f17 skybox, clean mocap, vivid block) without altering physics."""
    u = env.unwrapped
    m = u.model
    v = u.mujoco_renderer._get_viewer("rgb_array")

    if m.ntex > 0 and int(m.tex_type[0]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
        adr = int(m.tex_adr[0])
        h, w = int(m.tex_height[0]), int(m.tex_width[0])
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        sky = np.zeros((h, w, 3), dtype=np.uint8)
        sky[..., 0] = np.clip(10 + 13 * ys, 0, 255).astype(np.uint8)
        sky[..., 1] = np.clip(14 + 19 * ys, 0, 255).astype(np.uint8)
        sky[..., 2] = np.clip(22 + 30 * ys, 0, 255).astype(np.uint8)
        m.tex_data[adr : adr + h * w * 3] = sky.reshape(-1)
        if hasattr(v, "con") and v.con is not None:
            mujoco.mjr_uploadTexture(m, v.con, 0)

    # Hide green mocap crosshair axes (geoms 1..4)
    for idx in range(1, min(5, m.ngeom)):
        if m.geom_rgba[idx, 1] > 0.4 and m.geom_rgba[idx, 0] == 0.0:
            m.geom_rgba[idx, 3] = 0.0

    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name == "floor0":
            m.geom_rgba[i] = [0.055, 0.075, 0.11, 1.0]
            m.geom_matid[i] = -1
        elif name == "object0":
            m.geom_rgba[i] = [1.0, 0.56, 0.10, 1.0]
            m.geom_matid[i] = -1
        elif name is None and i == 22:
            m.geom_rgba[i] = [0.14, 0.18, 0.25, 1.0]
            m.geom_matid[i] = -1
        elif name and "gripper_finger" in name:
            m.geom_rgba[i] = [0.18, 0.85, 1.0, 1.0]

    for i in range(m.nsite):
        sname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
        if sname == "target0":
            m.site_rgba[i] = [0.20, 0.95, 0.62, 0.95]
        elif sname == "object0":
            m.site_rgba[i] = [0.0, 0.0, 0.0, 0.0]

    v.cam.distance = 1.24
    v.cam.elevation = -20.0
    v.cam.azimuth = 136.0
    v.cam.lookat[:] = [1.29, 0.74, 0.63]


def extract_camera_Basis(env, width: int, height: int) -> dict[str, np.ndarray | float]:
    """Capture exact monoscopic OpenGL camera parameters from MuJoCo's stereo mjvScene pair."""
    u = env.unwrapped
    v = u.mujoco_renderer._get_viewer("rgb_array")
    c0, c1 = v.scn.camera[0], v.scn.camera[1]
    pos = 0.5 * (np.array(c0.pos, dtype=np.float64) + np.array(c1.pos, dtype=np.float64))
    fwd = 0.5 * (np.array(c0.forward, dtype=np.float64) + np.array(c1.forward, dtype=np.float64))
    fwd /= max(np.linalg.norm(fwd), 1e-12)
    up_raw = 0.5 * (np.array(c0.up, dtype=np.float64) + np.array(c1.up, dtype=np.float64))
    right = np.cross(fwd, up_raw)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, fwd)
    fovy = np.deg2rad(float(u.model.vis.global_.fovy))
    fy = (height / 2.0) / math.tan(fovy / 2.0)
    return {"pos": pos, "fwd": fwd, "right": right, "up": up, "fx": fy, "fy": fy, "w": width, "h": height}


def project_3d(points_xyz: np.ndarray, cam: dict) -> np.ndarray:
    """Project (..., 3) world coordinates to (..., 2) pixel coordinates."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    rel = pts - cam["pos"]
    z = np.maximum(rel @ cam["fwd"], 1e-4)
    x = rel @ cam["right"]
    y = rel @ cam["up"]
    px = cam["w"] / 2.0 + cam["fx"] * (x / z)
    py = cam["h"] / 2.0 - cam["fy"] * (y / z)
    return np.stack([px, py], axis=-1)


# ---------------------------------------------------------------------------
# Closed-Loop Multi-Step Holographic Rollout (Zero Teacher Forcing)
# ---------------------------------------------------------------------------


@torch.no_grad()
def probe_3d_cloud_and_rollout(
    session: RoboticsSession,
    current_obs: dict,
    current_action: np.ndarray,
    ep_step: int,
    probe_rng: torch.Generator,
    device: str,
    horizon: int = ROLLOUT_HORIZON,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Query 32-particle 1-step cloud + 4-mode closed-loop multi-step trajectory ribbons.

    Executes entirely inside ``isolated_session(session)`` with ``batch_size=1`` per
    forward and a local CPU action RNG so training trajectories are 100% untouched.
    """
    z_dim = session.trainer.prior.z_dim
    zero_z = torch.zeros(1, z_dim, device=device)
    grip0 = current_obs["observation"][:3].astype(np.float64).copy()
    obj0 = current_obs["observation"][3:6].astype(np.float64).copy()

    # 1. All 32 particles' next-step predictions (each forward strictly batch_size=1)
    grip_cloud = np.zeros((32, 3), dtype=np.float64)
    obj_cloud = np.zeros((32, 3), dtype=np.float64)
    with isolated_session(session):
        for row in range(32):
            frame = _probe(session, row, zero_z, probe_rng)
            delta = rdata.decode_delta(frame, 6)[0].cpu().numpy()
            grip_cloud[row] = grip0 + delta[:3]
            obj_cloud[row] = obj0 + delta[3:6]

    # 2. Closed-loop multi-step rollout (zero simulator calls, zero teacher forcing)
    steps_ahead = max(1, min(horizon, rdata.EPISODE_LENGTH - ep_step))
    grip_ribbons: list[np.ndarray] = []
    obj_ribbons: list[np.ndarray] = []
    for row in (0, 8, 16, 24):
        with isolated_session(session):
            sim_obs = {
                "observation": current_obs["observation"].copy(),
                "desired_goal": current_obs["desired_goal"].copy(),
                "achieved_goal": current_obs["achieved_goal"].copy(),
            }
            local_rng = torch.Generator(device="cpu").manual_seed(1000 + ep_step * 37 + row)
            g_traj = [grip0.copy()]
            o_traj = [obj0.copy()]
            for h in range(steps_ahead):
                act_h = (
                    current_action
                    if h == 0
                    else rdata.policy_action("fetch_pick_and_place", sim_obs, ep_step + h, local_rng, 4)
                )
                if h > 0:
                    with prefix_context(session, list(rdata.encode_source(sim_obs, act_h, device))):
                        pred_frame = _probe(session, row, zero_z, probe_rng)
                else:
                    pred_frame = _probe(session, row, zero_z, probe_rng)
                d_xyz = rdata.decode_delta(pred_frame, 6)[0].cpu().numpy()
                g_next = np.clip(g_traj[-1] + d_xyz[:3], [0.8, 0.3, 0.38], [1.8, 1.2, 1.15])
                o_next = np.clip(o_traj[-1] + d_xyz[3:6], [0.8, 0.3, 0.38], [1.8, 1.2, 1.15])
                g_traj.append(g_next)
                o_traj.append(o_next)
                # Self-feed predicted positions into next step's state observation
                sim_obs["observation"][:3] = g_next
                sim_obs["observation"][3:6] = o_next
                sim_obs["observation"][6:9] = o_next - g_next
                sim_obs["achieved_goal"] = o_next.copy()
            grip_ribbons.append(np.stack(g_traj, axis=0))
            obj_ribbons.append(np.stack(o_traj, axis=0))

    return grip_cloud, obj_cloud, grip_ribbons, obj_ribbons


# ---------------------------------------------------------------------------
# Dark-Mode Dashboard & Holographic Glow Compositor
# ---------------------------------------------------------------------------


def load_fonts() -> dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    sans_candidates = [
        "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    ]
    mono_candidates = [
        "/usr/share/fonts/google-noto-vf/NotoSansMono[wght].ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
    ]
    sans_path = next((p for p in sans_candidates if Path(p).exists()), None)
    mono_path = next((p for p in mono_candidates if Path(p).exists()), None)
    if sans_path and mono_path:
        return {
            "title": ImageFont.truetype(sans_path, 16),
            "sub": ImageFont.truetype(sans_path, 12),
            "small": ImageFont.truetype(sans_path, 11),
            "mono": ImageFont.truetype(mono_path, 12),
            "mono_sm": ImageFont.truetype(mono_path, 11),
            "badge": ImageFont.truetype(mono_path, 12),
        }
    fallback = ImageFont.load_default()
    return {k: fallback for k in ("title", "sub", "small", "mono", "mono_sm", "badge")}


def render_hero_with_ghosts(
    hero_rgb: np.ndarray,
    cam: dict,
    grip_cloud: np.ndarray,
    obj_cloud: np.ndarray,
    grip_ribbons: list[np.ndarray],
    obj_ribbons: list[np.ndarray],
    true_grip_future: np.ndarray,
    true_obj_future: np.ndarray,
    goal_xyz: np.ndarray,
) -> Image.Image:
    """Composite glowing 32-particle clouds and 6-step closed-loop ribbons onto the 3D scene."""
    base = Image.fromarray(hero_rgb).convert("RGBA")
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sharp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dg = ImageDraw.Draw(glow)
    ds = ImageDraw.Draw(sharp)

    # Target goal reticle
    goal_px = project_3d(goal_xyz, cam)
    gx, gy = float(goal_px[0]), float(goal_px[1])
    dg.ellipse((gx - 10, gy - 10, gx + 10, gy + 10), outline=(52, 211, 153, 150), width=2)
    ds.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), outline=(52, 211, 153, 220), width=1)

    # True upcoming trajectory guide (subtle white/emerald reference)
    for fut, col in ((true_grip_future, (180, 235, 255, 135)), (true_obj_future, (255, 220, 165, 145))):
        if len(fut) >= 2:
            pts = [tuple(map(float, p)) for p in project_3d(fut, cam)]
            ds.line(pts, fill=col, width=1)
            ex, ey = pts[-1]
            ds.ellipse((ex - 2.5, ey - 2.5, ex + 2.5, ey + 2.5), outline=col, width=1)

    # Closed-loop 6-step holographic ribbons (4 particle modes)
    for rib in obj_ribbons:
        pts = [tuple(map(float, p)) for p in project_3d(rib, cam)]
        if len(pts) >= 2:
            dg.line(pts, fill=(255, 158, 44, 150), width=5)
            ds.line(pts, fill=(255, 196, 92, 225), width=2)
            tx, ty = pts[-1]
            ds.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), fill=(255, 214, 130, 235))

    for rib in grip_ribbons:
        pts = [tuple(map(float, p)) for p in project_3d(rib, cam)]
        if len(pts) >= 2:
            dg.line(pts, fill=(34, 211, 238, 150), width=5)
            ds.line(pts, fill=(125, 242, 255, 230), width=2)
            tx, ty = pts[-1]
            ds.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), fill=(186, 250, 255, 240))

    # 32-particle 1-step distributional cloud
    for p in project_3d(obj_cloud, cam):
        px, py = float(p[0]), float(p[1])
        dg.ellipse((px - 4.5, py - 4.5, px + 4.5, py + 4.5), fill=(251, 146, 60, 85))
        ds.ellipse((px - 2.0, py - 2.0, px + 2.0, py + 2.0), fill=(253, 186, 116, 210))

    for p in project_3d(grip_cloud, cam):
        px, py = float(p[0]), float(p[1])
        dg.ellipse((px - 4.5, py - 4.5, px + 4.5, py + 4.5), fill=(56, 189, 248, 85))
        ds.ellipse((px - 2.0, py - 2.0, px + 2.0, py + 2.0), fill=(165, 243, 252, 215))

    glow_blurred = glow.filter(ImageFilter.GaussianBlur(radius=3.2))
    out = Image.alpha_composite(base, glow_blurred)
    out = Image.alpha_composite(out, sharp)
    return out.convert("RGB")


def draw_sparkline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: list[float],
    current_step: int,
    total_steps: int,
    y_max: float,
    line_color: tuple[int, int, int],
    threshold: float | None = None,
    threshold_label: str = "",
    font=None,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(11, 15, 23), outline=(42, 51, 66), width=1)
    w, h = x1 - x0 - 12, y1 - y0 - 14
    ox, oy = x0 + 6, y0 + 7

    # Subtle grid lines
    for frac in (0.25, 0.5, 0.75):
        gy = int(oy + h * frac)
        draw.line((ox, gy, ox + w, gy), fill=(22, 28, 39), width=1)

    if threshold is not None and y_max > 0:
        ty = oy + h * (1.0 - min(1.0, max(0.0, threshold / y_max)))
        for sx in range(ox, ox + w, 8):
            draw.line((sx, ty, min(sx + 4, ox + w), ty), fill=(52, 211, 153), width=1)

    shown = values[:current_step]
    if not shown:
        return
    # Smooth with short causal EMA for readable visual trace alongside raw envelope
    ema = []
    acc = shown[0]
    for v in shown:
        acc = 0.82 * acc + 0.18 * v
        ema.append(acc)

    pts = []
    raw_pts = []
    for idx, (rv, ev) in enumerate(zip(shown, ema)):
        px = ox + w * (idx / max(1, total_steps - 1))
        py_e = oy + h * (1.0 - min(1.0, max(0.0, ev / y_max)))
        py_r = oy + h * (1.0 - min(1.0, max(0.0, rv / y_max)))
        pts.append((px, py_e))
        raw_pts.append((px, py_r))

    if len(raw_pts) >= 2:
        faint = tuple(max(20, int(c * 0.36)) for c in line_color)
        draw.line(raw_pts, fill=faint, width=1)
    if len(pts) >= 2:
        draw.line(pts, fill=line_color, width=2)
    cx, cy = pts[-1]
    draw.ellipse((cx - 3.5, cy - 3.5, cx + 3.5, cy + 3.5), fill=line_color, outline=(240, 248, 255))

    if threshold is not None and threshold_label and font is not None:
        tb = draw.textbbox((0, 0), threshold_label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        lx, ly = x1 - tw - 8, y0 + 5
        draw.rectangle((lx - 3, ly - 1, lx + tw + 3, ly + th + 2), fill=(11, 15, 23))
        draw.text((lx, ly), threshold_label, fill=(52, 211, 153), font=font)


def compose_dashboard_frame(
    record: dict,
    all_3d_errors_mm: list[float],
    all_rgb_rmses: list[float],
    total_steps: int,
    fonts: dict,
) -> Image.Image:
    """Assemble the complete 1080x560 dark-mode dashboard frame."""
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (13, 17, 23))
    draw = ImageDraw.Draw(canvas)

    step = record["step"]
    ep = record["episode"] + 1
    ep_step = record["ep_step"]

    # Header bar
    draw.rectangle((0, 0, CANVAS_W, 44), fill=(18, 24, 34))
    draw.line((0, 44, CANVAS_W, 44), fill=(40, 50, 68), width=1)
    draw.text((16, 12), "Particle", fill=(245, 248, 255), font=fonts["title"])
    p_w = draw.textbbox((16, 12), "Particle", font=fonts["title"])[2]
    draw.text((p_w + 1, 12), "Stream", fill=(96, 165, 250), font=fonts["title"])
    s_w = draw.textbbox((p_w + 1, 12), "Stream", font=fonts["title"])[2]
    draw.text(
        (s_w + 12, 15),
        "|  MuJoCo FetchPickAndPlace-v4  ·  Online World Model (GAIL-BCAP)",
        fill=(156, 169, 189),
        font=fonts["sub"],
    )

    # Right header badges (centered text inside pills)
    phase_name, phase_col = (
        ("APPROACH", (56, 189, 248))
        if ep_step < 8
        else ("GRASP", (251, 191, 36))
        if ep_step < 20
        else ("LIFT", (52, 211, 153))
    )
    b1, b2, b3 = (588, 9, 774, 35), (782, 9, 970, 35), (978, 9, 1066, 35)
    draw.rounded_rectangle(b1, radius=5, fill=(26, 33, 48), outline=(129, 140, 248), width=1)
    draw.text(((b1[0] + b1[2]) // 2, 22), "batch_size=1 · replay=0", fill=(199, 210, 254), font=fonts["badge"], anchor="mm")

    draw.rounded_rectangle(b2, radius=5, fill=(24, 32, 45), outline=(64, 82, 108), width=1)
    draw.text(((b2[0] + b2[2]) // 2, 22), f"Ep {ep:2d}/12 · Step {step:3d}/{total_steps}", fill=(226, 232, 240), font=fonts["badge"], anchor="mm")

    draw.rounded_rectangle(b3, radius=5, fill=(20, 29, 40), outline=phase_col, width=1)
    draw.text(((b3[0] + b3[2]) // 2, 22), phase_name, fill=phase_col, font=fonts["badge"], anchor="mm")

    # --- Panel 1: Left Hero 3D Scene (14, 56, 586, 546) ---
    p1 = (14, 56, 586, 546)
    draw.rounded_rectangle(p1, radius=8, fill=(17, 22, 31), outline=(42, 52, 68), width=1)
    act_titles = {
        0: "Cold Start (Random Weights — Exploring)",
        2: "Rapid Online Adaptation (80 Arrivals)",
        6: "Contact & Lift Dynamics Acquired",
        11: "Converged 3D Holographic Lookahead",
    }
    subtitle = act_titles.get(record["episode"], "Online Sequential Learning")
    draw.text((26, 65), "3D MuJoCo Scene + 32-Particle Closed-Loop Ghost Rollout", fill=(236, 242, 250), font=fonts["sub"])
    draw.text((26, 82), f"{subtitle}  ·  6-step lookahead without teacher forcing", fill=(138, 153, 176), font=fonts["small"])

    hero_img = render_hero_with_ghosts(
        record["hero_rgb"],
        record["cam"],
        record["grip_cloud"],
        record["obj_cloud"],
        record["grip_ribbons"],
        record["obj_ribbons"],
        record["true_grip_future"],
        record["true_obj_future"],
        record["goal_xyz"],
    )
    canvas.paste(hero_img, (24, 102))
    draw.rectangle((24, 102, 24 + HERO_W, 102 + HERO_H), outline=(48, 60, 80), width=1)

    # Legend under hero
    ly = 523
    draw.ellipse((28, ly + 3, 36, ly + 11), fill=(56, 189, 248))
    draw.text((41, ly), "32-Particle Gripper + 6-Step Arc", fill=(176, 190, 210), font=fonts["small"])
    draw.ellipse((238, ly + 3, 246, ly + 11), fill=(251, 146, 60))
    draw.text((251, ly), "32-Particle Block + Lift Arc", fill=(176, 190, 210), font=fonts["small"])
    draw.ellipse((426, ly + 3, 434, ly + 11), outline=(52, 211, 153), width=2)
    draw.text((439, ly), f"Goal  |  Err: {record['err_mm']:.2f} mm", fill=(152, 232, 193), font=fonts["small"])

    # --- Panel 2: Middle Live RGB Pixel Dreamer (598, 56, 838, 546) ---
    p2 = (598, 56, 838, 546)
    draw.rounded_rectangle(p2, radius=8, fill=(17, 22, 31), outline=(42, 52, 68), width=1)
    draw.text((610, 65), "RGB Pixel Dreamer (48×64)", fill=(236, 242, 250), font=fonts["sub"])
    draw.text((610, 82), "Predicted live before env.step()", fill=(138, 153, 176), font=fonts["small"])

    # Actual arriving camera frame (208x156)
    draw.text((614, 102), "Actual Next Camera Frame I_t", fill=(251, 191, 36), font=fonts["small"])
    t_pil = Image.fromarray(record["target_rgb"]).resize((208, 156), Image.Resampling.NEAREST)
    canvas.paste(t_pil, (614, 119))
    draw.rectangle((614, 119, 822, 275), outline=(217, 155, 38), width=1)

    # Dreamed RGB frame predicted BEFORE step!
    draw.text((614, 285), "Dreamed Pre-Step G_rgb(z, c_{t-1})", fill=(56, 189, 248), font=fonts["small"])
    p_pil = Image.fromarray(record["pred_rgb"]).resize((208, 156), Image.Resampling.NEAREST)
    canvas.paste(p_pil, (614, 302))
    border_col = (52, 211, 153) if record["rgb_rmse"] < 0.065 else (56, 189, 248)
    draw.rectangle((614, 302, 822, 458), outline=border_col, width=2)

    # 4-Particle visual imagination strip
    draw.text((614, 466), "4-Particle Visual Modes:", fill=(148, 163, 184), font=fonts["small"])
    for idx, mode_rgb in enumerate(record["mode_rgbs"]):
        mx = 614 + idx * 53
        m_pil = Image.fromarray(mode_rgb).resize((49, 37), Image.Resampling.NEAREST)
        canvas.paste(m_pil, (mx, 483))
        draw.rectangle((mx, 483, mx + 49, 520), outline=(56, 70, 92), width=1)
    draw.text((614, 525), f"Live RGB RMSE: {record['rgb_rmse']:.4f}", fill=(165, 243, 252), font=fonts["mono_sm"])

    # --- Panel 3: Right Online Convergence & Contract (850, 56, 1066, 546) ---
    p3 = (850, 56, 1066, 546)
    draw.rounded_rectangle(p3, radius=8, fill=(17, 22, 31), outline=(42, 52, 68), width=1)
    draw.text((862, 65), "Online Learning Curves", fill=(236, 242, 250), font=fonts["sub"])
    draw.text((862, 82), "Zero replay · 1 update / frame", fill=(138, 153, 176), font=fonts["small"])

    draw.text((862, 104), "3D Dynamics Error (mm)", fill=(125, 211, 252), font=fonts["small"])
    draw_sparkline(
        draw,
        (862, 121, 1054, 241),
        all_3d_errors_mm,
        step,
        total_steps,
        y_max=22.0,
        line_color=(56, 189, 248),
        threshold=5.0,
        threshold_label="<= 5.0 mm",
        font=fonts["small"],
    )

    draw.text((862, 253), "RGB Dreamer Error (RMSE)", fill=(192, 132, 252), font=fonts["small"])
    draw_sparkline(
        draw,
        (862, 270, 1054, 390),
        all_rgb_rmses,
        step,
        total_steps,
        y_max=0.35,
        line_color=(168, 85, 247),
        threshold=0.065,
        threshold_label="sharp <=.065",
        font=fonts["small"],
    )

    # Bottom strict online contract telemetry card
    card = (862, 402, 1054, 534)
    draw.rounded_rectangle(card, radius=6, fill=(11, 15, 23), outline=(42, 52, 68), width=1)
    lift_cm = max(0.0, (record["obj_z"] - record["obj_z0"]) * 100.0)
    ema_3d = float(np.mean(all_3d_errors_mm[max(0, step - 15) : step]))
    rows_txt = [
        ("Updates:", f"{step:3d} / {total_steps}", (226, 232, 240)),
        ("Batch (R/F):", "1 / 1", (129, 140, 248)),
        ("Replay Buf:", "0 frames", (52, 211, 153)),
        ("3D RMSE(15):", f"{ema_3d:5.2f} mm", (56, 189, 248)),
        ("Block Lift:", f"+{lift_cm:4.1f} cm", (251, 191, 36)),
    ]
    for r_i, (k_txt, v_txt, v_col) in enumerate(rows_txt):
        ry = 411 + r_i * 23
        draw.text((870, ry), k_txt, fill=(148, 163, 184), font=fonts["mono_sm"])
        draw.text((962, ry), v_txt, fill=v_col, font=fonts["mono_sm"])

    return canvas


# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------


def run_demo(
    *,
    output_gif: Path,
    output_mp4: Path | None = None,
    recipe_id: str = "online-gail-bcap-schedule-free",
    device: str = "cuda",
    steps: int = 480,
    fps: int = 12,
    showcase_episodes: tuple[int, ...] = SHOWCASE_EPISODES,
    frame_stride: int = 2,
) -> dict[str, float | int | str]:
    if device == "cuda" and torch.cuda.device_count() > 1:
        device = "cuda:1"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    ws = StreamWorkspace(ROOT, registry=REGISTRY)
    task = ws.load_tasks()["seq_robotics_fetch_pick_and_place"]
    candidates = ws.load_candidates()
    if recipe_id not in candidates:
        raise ValueError(f"Unknown recipe '{recipe_id}'. Available: {sorted(candidates)}")
    candidate = candidates[recipe_id]
    streams = NamedStreams(seed=0)

    # 1. Build 3D Particle Dynamics Learner (strictly batch_size=1, replay=0)
    trainer, g, d, memory, recipe = _build(task, candidate, streams, steps, device)
    session = RoboticsSession(
        trainer,
        g,
        d,
        memory,
        serving_generator=streams.generator("eval", component="serving", purpose="prediction", device=device),
    )

    # 2. Build Live RGB Pixel Dreamer (strictly batch_size=1, replay=0)
    rgb_streams = NamedStreams(seed=7)
    rgb_g = RGBStreamGenerator(z_dim=24, width=48, context_dim=29).to(device)
    rgb_d = RGBStreamCritic(width=24, context_dim=29).to(device)
    for mod, comp in ((rgb_g, "rgb_g"), (rgb_d, "rgb_d")):
        _init_module(mod, rgb_streams, comp)
    rgb_recipe = recipe.replace(
        z_dim=24, lr=0.0020, lr_schedule="cosine", lr_anneal_start=0.45, lr_floor=0.12, betas=(0.5, 0.999)
    )
    rgb_prior = rgb_recipe.replace(prior_update="learned").make_prior(
        device=device,
        dtype=torch.float32,
        init_std=1.0,
        sigma=0.025,
        generator=rgb_streams.generator("init", component="rgb_prior", purpose="locations", device=device),
    )
    _init_module(rgb_prior, rgb_streams, "rgb_prior")
    rgb_recipe.apply_prior_policy(rgb_prior)

    t_opts = dict(candidate.memory_policy.get("transport") or {})
    t_opts.pop("router", None)
    t_opts.update(
        {
            "context_bypass_weight": 0.0,
            "schedule_free_beta": 0.0,
            "style_bonus_weight": 0.001,
            "reconstruction_loss": "binary_cross_entropy",
            "reconstruction_weight": 55.0,
            "encoder_seed": rgb_streams.seed_for("init", component="rgb_encoder", purpose="projection"),
        }
    )
    rgb_trainer = TransportGANTrainer(
        rgb_recipe,
        rgb_g,
        rgb_d,
        prior=rgb_prior,
        seed=7,
        max_steps=steps,
        router_class=RGBGAILReconstruction,
        transport=t_opts,
        routing_generator=rgb_streams.generator("prior", component="rgb_routing", purpose="assignment", device=device),
        latent_generator=rgb_streams.generator("prior", component="rgb_latent", purpose="indices", device=device),
        penalty_generator=rgb_streams.generator("noise", component="rgb_penalty", purpose="training", device=device),
        noise_generator=rgb_streams.generator("noise", component="rgb_gen", purpose="output", device=device),
        input_noise_generator=rgb_streams.generator("noise", component="rgb_critic", purpose="input", device=device),
        eval_generator=rgb_streams.generator("eval", component="rgb_sampler", purpose="samples", device=device),
        model_generator=rgb_streams.generator("noise", component="rgb_models", purpose="stochastic", device=device),
        prior_noise_generator=rgb_streams.generator("noise", component="rgb_prior_n", purpose="gaussian", device=device),
        require_latent_damping=False,
    )
    rgb_eval_rng = rgb_streams.generator("eval", component="rgb_serving", purpose="pred", device=device)

    # Strict batch_size=1 forward guards on every network
    hooks = []
    for module in (g, d, rgb_g, rgb_d):
        hooks.append(
            module.register_forward_pre_hook(
                lambda _m, args: (_ for _ in ()).throw(ValueError("strict batch_size=1 violated"))
                if len(args[0]) != 1
                else None
            )
        )

    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", width=HERO_W, height=HERO_H)
    data_rng = streams.generator("data", component="robotics", purpose="training_episodes", device="cpu")
    probe_rng = streams.generator("eval", component="robotics", purpose="probes", device=device)

    all_3d_errors_mm: list[float] = []
    all_rgb_rmses: list[float] = []
    episode_trajectories: dict[int, dict[str, list[np.ndarray]]] = {}
    captured_records: list[dict] = []

    zero_rgb_z = torch.zeros(1, 24, device=device)
    styled = False
    cam_basis = None
    ep_obj_z0 = 0.42
    t0 = time.perf_counter()

    try:
        for step in range(1, steps + 1):
            episode = (step - 1) // rdata.EPISODE_LENGTH
            ep_step = (step - 1) % rdata.EPISODE_LENGTH
            if ep_step == 0:
                current, _ = rdata.reset_env(env, data_rng)
                if not styled:
                    env.render()
                    style_mujoco_scene(env)
                    env.render()
                    cam_basis = extract_camera_Basis(env, HERO_W, HERO_H)
                    styled = True
                ep_obj_z0 = float(current["observation"][5])
                episode_trajectories[episode] = {
                    "grip": [current["observation"][:3].astype(np.float64).copy()],
                    "obj": [current["observation"][3:6].astype(np.float64).copy()],
                }

            action = rdata.policy_action("fetch_pick_and_place", current, ep_step, data_rng, env.action_space.shape[0])
            session.source_pair(*rdata.encode_source(current, action, device))
            rgb_ctx = build_rgb_context(session.context, current, action, device)

            # Live pre-step predictions BEFORE simulator steps!
            pred_3d = session.predict()
            rgb_g.set_context(rgb_ctx)
            rgb_d.set_context(rgb_ctx)
            with torch.no_grad():
                rgb_g.eval()
                rgb_trainer.prior.eval()
                z_rgb, _ = rgb_trainer.prior.sample(
                    1, fixed_first_n=True, offset=0, eps=zero_rgb_z, generator=rgb_eval_rng
                )
                pred_rgb = rgb_g(z_rgb)

            is_showcase_frame = (episode in showcase_episodes) and (
                ep_step % frame_stride == 0 or ep_step == rdata.EPISODE_LENGTH - 1
            )
            if is_showcase_frame:
                grip_cloud, obj_cloud, grip_ribbons, obj_ribbons = probe_3d_cloud_and_rollout(
                    session, current, action, ep_step, probe_rng, device, horizon=ROLLOUT_HORIZON
                )
                mode_rgbs = []
                with torch.no_grad():
                    rgb_g.eval()
                    rgb_trainer.prior.eval()
                    for row in (0, 8, 16, 24):
                        zm, _ = rgb_trainer.prior.sample(
                            1, fixed_first_n=True, offset=row, eps=zero_rgb_z, generator=rgb_eval_rng
                        )
                        im = (rgb_g(zm)[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        mode_rgbs.append(im)

            # Advance real MuJoCo physics by 1 step
            following, _, _, _, _ = env.step(action)
            hero_rgb = env.render().copy()
            target_rgb = F.interpolate(
                torch.from_numpy(hero_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0,
                size=(DREAM_H, DREAM_W),
                mode="area",
            ).detach()

            episode_trajectories[episode]["grip"].append(following["observation"][:3].astype(np.float64).copy())
            episode_trajectories[episode]["obj"].append(following["observation"][3:6].astype(np.float64).copy())

            delta = rdata.tracked_positions("fetch_pick_and_place", following) - rdata.tracked_positions(
                "fetch_pick_and_place", current
            )
            err_mm = (
                float((rdata.decode_delta(pred_3d, 6).double() - torch.tensor(delta, device=device)).square().mean().sqrt())
                * 1000.0
            )
            rgb_rmse = float((pred_rgb - target_rgb).square().mean().sqrt())
            all_3d_errors_mm.append(err_mm)
            all_rgb_rmses.append(rgb_rmse)

            # Single-arrival online updates (batch_size=1, replay=0)
            session.observe_target(rdata.encode_delta(delta, device))
            rgb_g.train()
            rgb_d.train()
            rgb_trainer.prior.train()
            rgb_trainer.step(target_rgb)

            if is_showcase_frame:
                t_np = (target_rgb[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                p_np = (pred_rgb[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                captured_records.append(
                    {
                        "step": step,
                        "episode": episode,
                        "ep_step": ep_step,
                        "hero_rgb": hero_rgb,
                        "cam": cam_basis,
                        "grip_cloud": grip_cloud,
                        "obj_cloud": obj_cloud,
                        "grip_ribbons": grip_ribbons,
                        "obj_ribbons": obj_ribbons,
                        "goal_xyz": following["desired_goal"].astype(np.float64).copy(),
                        "target_rgb": t_np,
                        "pred_rgb": p_np,
                        "mode_rgbs": mode_rgbs,
                        "err_mm": err_mm,
                        "rgb_rmse": rgb_rmse,
                        "obj_z": float(following["observation"][5]),
                        "obj_z0": ep_obj_z0,
                    }
                )

            current = following
    finally:
        env.close()
        for h in hooks:
            h.remove()

    assert session.max_real_batch == 1 and session.max_fake_batch == 1
    assert session.diagnostics()["replayed_examples"] == 0

    # Attach ground-truth future trajectories (for visual reference only, post-hoc)
    fonts = load_fonts()
    rendered_frames: list[Image.Image] = []
    durations_ms: list[int] = []
    base_ms = max(40, int(round(1000 / fps)))

    for i, rec in enumerate(captured_records):
        ep = rec["episode"]
        es = rec["ep_step"]
        end_idx = min(rdata.EPISODE_LENGTH, es + ROLLOUT_HORIZON) + 1
        rec["true_grip_future"] = np.stack(episode_trajectories[ep]["grip"][es:end_idx], axis=0)
        rec["true_obj_future"] = np.stack(episode_trajectories[ep]["obj"][es:end_idx], axis=0)
        frame_img = compose_dashboard_frame(rec, all_3d_errors_mm, all_rgb_rmses, steps, fonts)
        rendered_frames.append(frame_img)
        # Brief pause at the end of each episode act, slightly longer on the final frame
        is_act_end = (i == len(captured_records) - 1) or (captured_records[i + 1]["episode"] != ep)
        durations_ms.append(base_ms * 7 if i == len(captured_records) - 1 else base_ms * 3 if is_act_end else base_ms)

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    # Build a shared global 256-color palette from representative early + late frames for compact delta compression
    sample_strip = Image.new("RGB", (CANVAS_W * 2, CANVAS_H))
    sample_strip.paste(rendered_frames[0], (0, 0))
    sample_strip.paste(rendered_frames[-1], (CANVAS_W, 0))
    master_palette = sample_strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantized = [f.quantize(palette=master_palette, dither=Image.Dither.NONE) for f in rendered_frames]
    quantized[0].save(
        output_gif,
        save_all=True,
        append_images=quantized[1:],
        duration=durations_ms,
        loop=0,
        optimize=True,
    )

    if output_mp4 is not None:
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{CANVAS_W}x{CANVAS_H}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output_mp4),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for idx, frame_img in enumerate(rendered_frames):
            repeats = 5 if idx == len(rendered_frames) - 1 else 2 if durations_ms[idx] > base_ms else 1
            raw = frame_img.tobytes()
            for _ in range(repeats):
                proc.stdin.write(raw)
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")

    elapsed = time.perf_counter() - t0
    return {
        "recipe": recipe_id,
        "steps": steps,
        "rendered_frames": len(rendered_frames),
        "initial_3d_err_mm": round(float(np.mean(all_3d_errors_mm[:15])), 3),
        "final_3d_err_mm": round(float(np.mean(all_3d_errors_mm[-40:])), 3),
        "initial_rgb_rmse": round(float(np.mean(all_rgb_rmses[:5])), 4),
        "final_rgb_rmse": round(float(np.mean(all_rgb_rmses[-40:])), 4),
        "max_real_batch": session.max_real_batch,
        "max_fake_batch": session.max_fake_batch,
        "replayed_examples": session.diagnostics()["replayed_examples"],
        "output_gif": str(output_gif),
        "output_mp4": str(output_mp4) if output_mp4 else "",
        "elapsed_seconds": round(elapsed, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/media/demos/mujoco-pick-and-place.gif",
        help="Output animated GIF path",
    )
    parser.add_argument(
        "--mp4",
        type=Path,
        default=None,
        help="Optional H.264 MP4 output path",
    )
    parser.add_argument("--recipe", default="online-gail-bcap-schedule-free")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args(argv)

    summary = run_demo(
        output_gif=args.output,
        output_mp4=args.mp4,
        recipe_id=args.recipe,
        device=args.device,
        steps=args.steps,
        fps=args.fps,
    )
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

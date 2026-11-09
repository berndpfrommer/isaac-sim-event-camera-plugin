"""
single_cam.py
=============
Render a white square rotating counterclockwise against black. It completes one
revolution in five simulation seconds. Keyframes at ``render_hz`` are warped
``--warp``×, converted to DVS events, and written to ``--dir``:

    env0_ep0.h5        DVS events (groups DVS/cam0; x,y,t,p)
    rgb_stereo.mp4     synthesised RGB, cam0
    events.mp4         the DVS events, cam0

The whole DVS pipeline is the four ``dvs.*`` calls below — everything else is just
stepping the sim and writing videos.
"""

import argparse
import math
import os

parser = argparse.ArgumentParser(
    description="dvs_gen quickstart: sim -> warp -> events -> video"
)
parser.add_argument(
    "--render_hz", type=float, default=125, help="keyframe render rate (= sim dt)"
)
parser.add_argument(
    "--warp", type=int, default=8, help="warp multiplier K (events at render_hz*K)"
)
parser.add_argument(
    "--keyframes", type=int, default=None,
    help="number of keyframes to simulate (default: five seconds at render_hz)",
)
parser.add_argument("--dir", type=str, default="/tmp/dvs_quickstart")
parser.add_argument(
    "--margin",
    type=int,
    nargs="+",
    default=[0],
    help="over-render this many pixels per side then crop them off "
    "(1 value = all sides; 4 values = left right top bottom)",
)
parser.add_argument(
    "--event_source",
    default=None,
    choices=["ldr", "hdr"],
    help="OVERRIDE the config's camera event_source. Default: follow the config "
    "(set DVSCameraCfg.event_source='hdr' to run the whole pipeline on the "
    "linear HDR buffer; RGB video tone-mapped for display).",
)

from isaaclab.app import AppLauncher

import torch

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not math.isfinite(args.render_hz) or args.render_hz <= 0:
    parser.error("--render_hz must be positive and finite")
if args.warp <= 0:
    parser.error("--warp must be positive")
if args.keyframes is not None and args.keyframes <= 0:
    parser.error("--keyframes must be positive")
if args.keyframes is None:
    args.keyframes = math.ceil(5.0 * args.render_hz)
# Open the Kit viewport window unless the user picked a visualizer / --headless.
if not getattr(args, "headless", False) and not getattr(args, "visualizer", None):
    args.visualizer = ["kit"]
app_launcher = AppLauncher(args, enable_cameras=True)
simulation_app = app_launcher.app

import cv2
import h5py
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from dvs_gen.env import DVSEnvCfg
from dvs_gen.sensors import DVSCamera, parse_margin
from dvs_gen.io import H264Writer

K = args.warp  # warp multiplier
DT_KEY = 1.0 / args.render_hz
DT_FINE = DT_KEY / K
OUT_HZ = args.render_hz * K  # effective event rate (for video fps)
PLAYBACK_HZ = max(1, int(round(OUT_HZ / 20)))
ROTATION_PERIOD_S = 5.0


def configure_rotating_shape(cfg):
    """Replace only the shape spawner to use another rigid geometric primitive."""
    cfg.scene.cam0.offset.pos = (0.0, -0.7, 1.8)
    cfg.scene.cam0.offset.rot = (2 ** -0.5, 0.0, 0.0, 2 ** -0.5)
    cfg.scene.cam0.offset.convention = "opengl"
    shape_spawn = sim_utils.CuboidCfg(
        size=(0.6, 0.01, 0.6),
        rigid_props=PhysxRigidBodyCfg(disable_gravity=True, angular_damping=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 1.0, 1.0),
            emissive_color=(1.0, 1.0, 1.0),
            roughness=1.0,
        ),
    )
    cfg.scene.DroppedObject = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DroppedObject",
        spawn=shape_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.8, 1.8),
            # The camera looks along +Y, so -Y is counterclockwise in its image.
            ang_vel=(0.0, -2.0 * math.pi / ROTATION_PERIOD_S, 0.0),
        ),
    )
    cfg.scene.dome_light = AssetBaseCfg(
        prim_path="/World/BackgroundDomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, visible_in_primary_ray=False),
    )
    cfg.events.randomize_background = None
    cfg.events.drop_object = None
    cfg.terminations.object_dropped = None


def render_event_video(
    h5_path,
    out_path,
    cams=("cam0",),
    H=480,
    W=640,
    fps=50,
    sample_hz=None,
    win=None,
):
    """render events(red = ON, blue = OFF) -> mp4.

    ``sample_hz`` = frame sampling rate in SIM time (default: same as ``fps`` =
    real-time playback); ``fps`` = playback rate of the mp4. Passing the RGB
    video's sampling/playback rates yields an event video frame-for-frame in
    sync with it. ``win`` = accumulation window (default: one sample period).
    """
    sample_hz = fps if sample_hz is None else sample_hz
    win = 1.0 / sample_hz if win is None else win
    with h5py.File(h5_path, "r") as f:
        data = {
            c: (
                f[f"DVS/{c}"]["x"][:],
                f[f"DVS/{c}"]["y"][:],
                f[f"DVS/{c}"]["t"][:].astype(np.float64),
                f[f"DVS/{c}"]["p"][:],
            )
            for c in cams
            if f"DVS/{c}" in f
        }
    tmin = min(d[2].min() for d in data.values())
    tmax = max(d[2].max() for d in data.values())
    print("found events in range: {} - {}".format(tmin, tmax))
    vw = H264Writer(out_path, W * len(data), H, fps)
    for tc in np.arange(tmin, tmax, 1.0 / sample_hz):
        panes = []
        for c in data:
            x, y, t, p = data[c]
            m = (t >= tc) & (t < tc + win)
            img = np.full((H, W, 3), 255, np.uint8)
            img[y[m][p[m] > 0], x[m][p[m] > 0]] = (255, 0, 0)  # ON  -> red
            img[y[m][p[m] <= 0], x[m][p[m] <= 0]] = (0, 0, 255)  # OFF -> blue
            panes.append(img)
        vw.write(cv2.cvtColor(np.concatenate(panes, axis=1), cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    os.makedirs(args.dir, exist_ok=True)

    # 1. Build the single-camera env, retuned to the keyframe rate.
    cfg = DVSEnvCfg()
    configure_rotating_shape(cfg)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.sim.dt = DT_KEY
    cfg.sim.render_interval = 1
    # Keep the rendered primitive in sync with its physics transform.
    cfg.sim.use_fabric = False
    # Account for the prime step and avoid auto-reset at the last keyframe.
    cfg.episode_length_s = (args.keyframes + 2) * DT_KEY
    # Over-render `margin` pixels per side; they get cropped off so the saved
    # RGB/events are artifact-free at the borders (principal point is shifted so
    # the cropped W0xH0 view matches the calibration exactly).
    margin = parse_margin(args.margin)
    L, R, T, Bm = margin
    W0, H0 = cfg.scene.cam0.width, cfg.scene.cam0.height
    for f in ("cam0",):
        c = getattr(cfg.scene, f)
        c.update_period = DT_KEY
        c.width = W0 + L + R
        c.height = H0 + T + Bm
    cfg.events.reinit_dvs.params["margin"] = margin
    # config-driven: DVSCameraCfg(event_source='hdr') auto-requests rgb_hdr and
    # from_scene picks it up. --event_source overrides the config.
    if args.event_source is not None:
        for f in ("cam0",):
            getattr(cfg.scene, f).event_source = args.event_source
    if any(
        getattr(getattr(cfg.scene, f), "event_source", "ldr") == "hdr"
        for f in ("cam0",)
    ):
        for f in ("cam0",):
            c = getattr(cfg.scene, f)
            if "rgb_hdr" not in c.data_types:
                c.data_types = list(c.data_types) + ["rgb_hdr"]
    env = ManagerBasedRLEnv(cfg=cfg)

    # 2. One object bundles the stereo cameras + event processors + recorder.
    dvs = DVSCamera.from_scene(
        env.scene,
        [
            "cam0",
        ],
        out_dir=args.dir,
        margin=margin,
    )

    # Stereo RGB video writer: cam0 (20x slow-mo to be watchable).
    W, H = W0, H0  # cropped (output) size
    vw_rgb = H264Writer(
        os.path.join(args.dir, "rgb_stereo.mp4"), W, H, PLAYBACK_HZ
    )

    def frame_cb(_i, frames):
        panes = [
            dvs.display_u8(frames[c][0]) for c in ("cam0",)
        ]  # tone-maps HDR for display
        vw_rgb.write(cv2.cvtColor(np.concatenate(panes, axis=1), cv2.COLOR_RGB2BGR))

    actions = torch.zeros((env.num_envs, 0), device=env.device)
    env.reset()
    env.scene["cam0"].set_world_poses_from_view(
        eyes=torch.tensor([[0.0, -0.7, 1.8]], device=env.device),
        targets=torch.tensor([[0.0, 0.8, 1.8]], device=env.device),
    )
    shape = env.scene["DroppedObject"]
    shape.write_root_velocity_to_sim(shape.data.default_root_state[:, 7:13].clone())
    # Headless Kit may otherwise skip its app loop, leaving RTX frames stale.
    if getattr(args, "headless", False):
        for _ in range(3):
            simulation_app.update()
    env.step(actions)  # prime step
    if getattr(args, "headless", False):
        simulation_app.update()
    prev = dvs.snapshot()
    t_prev = float(env.common_step_counter * env.step_dt)

    # 3. Main loop: step -> warp the gap -> emit events (all in warp_and_process).
    for _ in range(args.keyframes):
        _, _, terminated, truncated, _ = env.step(actions)
        if getattr(args, "headless", False):
            simulation_app.update()
        cur = dvs.snapshot()
        if bool(torch.logical_or(terminated, truncated).any()):
            # Never warp across an automatic reset.
            print(
                "[quickstart] episode ended — stopping recording",
                flush=True,
            )
            break
        dvs.warp_and_process(prev, cur, K, t_prev, DT_FINE, frame_cb=frame_cb)
        prev = cur
        t_prev = float(env.common_step_counter * env.step_dt)

    # 4. Flush this (env, episode) to HDF5.
    print("[quickstart] flushing episode to HDF5")
    dvs.flush(env_id=0, episode_idx=0)
    vw_rgb.release()

    h5_path = os.path.join(args.dir, "env0_ep0.h5")
    # Same sampling (one frame per fine frame) and playback rate as the RGB video,
    # so rgb_stereo.mp4 and events_stereo.mp4 run frame-for-frame in sync.
    render_event_video(
        h5_path,
        os.path.join(args.dir, "events.mp4"),
        H=H,
        W=W,
        fps=PLAYBACK_HZ,
        sample_hz=OUT_HZ,
    )
    print(
        f"[quickstart] done -> {args.dir}\n"
        f"  events : {h5_path}  (groups DVS/cam0)\n"
        f"  rgb    : rgb_stereo.mp4     (cam0)\n"
        f"  events : events.mp4         (cam0)",
        flush=True,
    )
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()

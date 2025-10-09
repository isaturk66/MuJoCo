# simulation.py
# Gymnasium-compatible MuJoCo drone environment with inner-loop PID control.
# - No waypointing
# - No outer PID / planner
# - LiDAR visualization only (not included in obs)
# - Actions drive inner-loop setpoints (safer than raw thrusts)
#
# Requirements:
#   pip install gymnasium numpy mujoco simple-pid
#
# Assumptions:
# - The MuJoCo model path below is valid in your repo:
#     "mujoco_menagerie-main/skydio_x2/scene.xml"
# - The base pose in qpos uses Euler angles after the first three xyz
#   or your previous model already matched this assumption. If your model stores
#   a quaternion instead, adapt `get_angles_from_state()` as needed.

from typing import Optional, Tuple
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from simple_pid import PID


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class DroneEnv(gym.Env):
    """
    A MuJoCo drone environment exposing a Gymnasium API (Application Programming Interface).
    Actions adjust inner-loop setpoints:
        action = [d_roll_sp, d_pitch_sp, yaw_rate_cmd, alt_rate_cmd]
    The inner PIDs (proportional–integral–derivative controllers) translate
    these setpoints/commands to motor thrusts each step.

    Observations (12D):
        [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    LiDAR (light detection and ranging) rays are visualized, but NOT returned in obs (yet).
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self,
                 model_path: str = "mujoco_menagerie-main/skydio_x2/scene.xml",
                 max_episode_steps: int = 2000,
                 init_hover_alt: float = 1.0,
                 render_mode: Optional[str] = None):
        super().__init__()

        # ---- MuJoCo model/data ----
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)

        # ---- Episode / render config ----
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self.init_hover_alt = init_hover_alt
        self.render_mode = render_mode
        self.viewer = None  # set on first render / external launcher

        # ---- LiDAR setup (visualization-only) ----
        self.lidar_sensors = [
            # Horizontal ring
            'lidar_0', 'lidar_45', 'lidar_90', 'lidar_135',
            'lidar_180', 'lidar_225', 'lidar_270', 'lidar_315',
            # Upper ring
            'lidar_up_0', 'lidar_up_45', 'lidar_up_90', 'lidar_up_135',
            'lidar_up_180', 'lidar_up_225', 'lidar_up_270', 'lidar_up_315',
            # Lower ring
            'lidar_down_0', 'lidar_down_45', 'lidar_down_90', 'lidar_down_135',
            'lidar_down_180', 'lidar_down_225', 'lidar_down_270', 'lidar_down_315',
            # Zenith / nadir
            'lidar_zenith', 'lidar_nadir'
        ]
        self._lidar_sensor_ids = []
        self._lidar_sensor_adr = []
        self._lidar_site_ids = []
        for name in self.lidar_sensors:
            sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self._lidar_sensor_ids.append(sid)
            self._lidar_sensor_adr.append(self.m.sensor_adr[sid])
            self._lidar_site_ids.append(
                mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)
            )

        # ---- Inner-loop controllers ----
        # (Gains from your working experiment; tune as needed.)
        self.pid_alt = PID(5.50844, 0.57871, 1.2, setpoint=0.0)  # outputs thrust term
        self.pid_roll = PID(2.6785, 0.56871, 1.2508, setpoint=0.0, output_limits=(-1, 1))
        self.pid_pitch = PID(2.6785, 0.56871, 1.2508, setpoint=0.0, output_limits=(-1, 1))
        self.pid_yaw = PID(0.54, 0.0, 5.358333, setpoint=1.0, output_limits=(-3, 3))

        # ---- Setpoints commanded by actions (internal state, not part of obs) ----
        self.roll_sp = 0.0
        self.pitch_sp = 0.0
        self.yaw_rate_cmd = 0.0
        self.alt_rate_cmd = 0.0
        self._base_hover_bias = 3.2495  # hover offset added to thrust PID

        # Neutral-decay toward hover when idle (Hz, exponential)
        self._sp_decay_hz = 3.0

        # ---- Action/Observation spaces ----
        # Action is small per-step delta for roll/pitch setpoint, plus direct rates for yaw/alt
        self._max_roll_step = np.deg2rad(2.0)
        self._max_pitch_step = np.deg2rad(2.0)
        self._max_yaw_rate = np.deg2rad(40.0)
        self._max_alt_rate = 1.0  # m/s
        self.action_space = spaces.Box(
            low=np.array([-self._max_roll_step, -self._max_pitch_step, -self._max_yaw_rate, -self._max_alt_rate], dtype=np.float32),
            high=np.array([ self._max_roll_step,  self._max_pitch_step,  self._max_yaw_rate,  self._max_alt_rate], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        high = np.array([np.inf]*12, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Optional: external key callback passthrough if someone wants to set it
        self._key_callback = None

    # -------------------------- Utility getters ---------------------------

    def _get_pos(self) -> np.ndarray:
        # Assumes first 3 qpos are world position
        return self.d.qpos[0:3].copy()

    def _get_lin_vel(self) -> np.ndarray:
        return self.d.qvel[0:3].copy()

    def _get_ang_vel(self) -> np.ndarray:
        # Next 3 in qvel are angular velocities for a free joint
        return self.d.qvel[3:6].copy()

    def _get_angles_from_state(self) -> Tuple[float, float, float]:
        """
        Extract (roll, pitch, yaw) from state.
        NOTE: This assumes your model's qpos stores Euler angles starting at index 3.
        If your base orientation is a quaternion, replace this with a quat->euler conversion.
        """
        # Your earlier working order looked like [roll, yaw, pitch]; remap to (roll, pitch, yaw)
        roll = float(self.d.qpos[3])
        yaw = float(self.d.qpos[4])
        pitch = float(self.d.qpos[5])
        return roll, pitch, yaw

    def _compute_motor_control(self, thrust, roll, pitch, yaw):
        # 4-motor mixing (same as your working mixer)
        return np.array([
            thrust + roll + pitch - yaw,
            thrust - roll + pitch + yaw,
            thrust - roll - pitch - yaw,
            thrust + roll - pitch + yaw
        ], dtype=np.float32)

    def _update_inner_control(self, dt: float):
        # Read current state
        pos = self._get_pos()
        roll, pitch, yaw = self._get_angles_from_state()
        alt = pos[2]

        # Update altitude setpoint via alt_rate_cmd
        self.pid_alt.setpoint += self.alt_rate_cmd * dt  # integrates altitude rate into altitude setpoint

        # Apply current setpoints to attitude PIDs
        self.pid_roll.setpoint = self.roll_sp
        self.pid_pitch.setpoint = self.pitch_sp
        # yaw PID used as stabilizer around setpoint=1; we add yaw_rate_cmd directly

        # PID outputs
        # NOTE: The PID names are misleading - this matches the working example.py behavior:
        # "pid_roll" actually controls based on YAW, "pid_yaw" controls based on ROLL
        cmd_thrust = self.pid_alt(alt) + self._base_hover_bias
        cmd_roll = - self.pid_roll(yaw)   # Use yaw for roll command (matches example.py)
        cmd_pitch = self.pid_pitch(pitch)
        cmd_yaw = - self.pid_yaw(roll) + self.yaw_rate_cmd  # Use roll for yaw command (matches example.py)

        # Send to actuators
        self.d.ctrl[:4] = self._compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)

    def _get_lidar_visuals(self):
        """Returns a list of (site_pos, hit_point, has_hit) for visualization."""
        out = []
        for i, adr in enumerate(self._lidar_sensor_adr):
            distance = self.d.sensordata[adr]
            site_id = self._lidar_site_ids[i]
            site_pos = self.d.site_xpos[site_id].copy()
            site_mat = self.d.site_xmat[site_id].reshape(3, 3)
            ray_dir = site_mat[:, 2]
            if 0.0 <= distance < 5.0:
                hit = site_pos + ray_dir * distance
                out.append((site_pos, hit, True))
            else:
                hit = site_pos + ray_dir * 5.0
                out.append((site_pos, hit, False))
        return out

    def _draw_lidar(self):
        """Draw LiDAR rays in the viewer (visual only)."""
        if self.viewer is None:
            return
        vis = self._get_lidar_visuals()
        with self.viewer.lock():
            # Toggle contact points for some visual motion
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.d.time % 2)
            # Clear previous custom geoms
            self.viewer.user_scn.ngeom = 0
            for site_pos, hit, has_hit in vis:
                if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom - 2:
                    break
                # Ray line
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 1, 0, 0.6]) if has_hit else np.array([0.6, 0.6, 0.6, 0.3])
                )
                mujoco.mjv_connector(
                    self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    2.0,
                    site_pos,
                    hit
                )
                self.viewer.user_scn.ngeom += 1

                # Hit sphere (only when there's an actual hit)
                if has_hit and self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                    mujoco.mjv_initGeom(
                        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.03, 0, 0]),
                        pos=hit,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0.2, 0, 0.8])
                    )
                    self.viewer.user_scn.ngeom += 1

    # -------------------------- Gymnasium API -----------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.m, self.d)

        # Initialize pose roughly at origin; set hover altitude setpoint
        self.pid_alt.setpoint = self.init_hover_alt
        self.roll_sp = 0.0
        self.pitch_sp = 0.0
        self.yaw_rate_cmd = 0.0
        self.alt_rate_cmd = 0.0
        self._elapsed_steps = 0

        # Step once to settle
        mujoco.mj_forward(self.m, self.d)

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        pos = self._get_pos()
        roll, pitch, yaw = self._get_angles_from_state()
        lin = self._get_lin_vel()
        ang = self._get_ang_vel()
        obs = np.array([pos[0], pos[1], pos[2],
                        roll, pitch, yaw,
                        lin[0], lin[1], lin[2],
                        ang[0], ang[1], ang[2]], dtype=np.float32)
        return obs

    def step(self, action: np.ndarray):
        self._elapsed_steps += 1
        dt = self.m.opt.timestep

        # ----- interpret action (deltas + rates)
        d_roll_sp, d_pitch_sp, yaw_rate_cmd, alt_rate_cmd = np.asarray(action, dtype=np.float32)

        # integrate roll/pitch setpoints by small deltas per step
        self.roll_sp = clamp(self.roll_sp + float(d_roll_sp), -np.deg2rad(30), np.deg2rad(30))
        self.pitch_sp = clamp(self.pitch_sp + float(d_pitch_sp), -np.deg2rad(30), np.deg2rad(30))

        # set rate commands directly for this step
        self.yaw_rate_cmd = float(yaw_rate_cmd)
        self.alt_rate_cmd = float(alt_rate_cmd)

        # ----- neutral decay so it hovers when there’s no input
        decay = math.exp(-self._sp_decay_hz * dt)
        self.roll_sp *= decay
        self.pitch_sp *= decay
        self.yaw_rate_cmd *= decay
        self.alt_rate_cmd *= decay

        # inner control → motors
        self._update_inner_control(dt)

        # physics step
        mujoco.mj_step(self.m, self.d)

        # render & LiDAR draw
        if self.render_mode == "human":
            if self.viewer is None:
                # If an external script set a key callback, pass it through
                kwargs = {}
                if self._key_callback is not None:
                    kwargs["key_callback"] = self._key_callback
                self.viewer = mujoco.viewer.launch_passive(self.m, self.d, **kwargs)
            self._draw_lidar()
            self.viewer.sync()

        # Observations, rewards, termination
        obs = self._get_obs()
        reward = 0.0  # placeholder for RL
        terminated = False
        truncated = self._elapsed_steps >= self.max_episode_steps
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                kwargs = {}
                if self._key_callback is not None:
                    kwargs["key_callback"] = self._key_callback
                self.viewer = mujoco.viewer.launch_passive(self.m, self.d, **kwargs)

    def set_key_callback(self, fn):
        """Optional: allow external scripts to register a key callback before render()."""
        self._key_callback = fn

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


# Optional: quick smoke test
if __name__ == "__main__":
    env = DroneEnv(render_mode="human")
    obs, info = env.reset()
    print("Env ready. Use manual_control.py for keyboard control.")
    for _ in range(120):
        a = np.zeros(4, dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    env.close()

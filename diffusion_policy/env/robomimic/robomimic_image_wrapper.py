from typing import Optional
import os
import numpy as np
import gym
from gym import spaces
from robomimic.envs.env_robosuite import EnvRobosuite


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key='agentview_image',
    ):
        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        # Occlusion switches (off by default)
        self.occlude_agentview = os.environ.get("DP_OCCLUDE_AGENTVIEW", "0") == "1"
        self.occluder_w = int(os.environ.get("DP_OCCLUDER_W", "44"))
        self.occluder_h = int(os.environ.get("DP_OCCLUDER_H", "88"))

        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")

            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    @staticmethod
    def _to_hwc_uint8(img):
        # support CHW float[0,1] or HWC uint8/float
        if img.ndim != 3:
            raise RuntimeError(f"Unsupported image ndim: {img.ndim}")

        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            hwc = np.moveaxis(img, 0, -1)
            layout = "chw"
        else:
            hwc = img
            layout = "hwc"

        if hwc.dtype == np.uint8:
            hwc_u8 = hwc.copy()
            dtype_kind = "uint8"
        else:
            hwc_u8 = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
            dtype_kind = "float01"

        return hwc_u8, layout, dtype_kind

    @staticmethod
    def _from_hwc_uint8(hwc_u8, layout, dtype_kind):
        out = hwc_u8
        if dtype_kind == "float01":
            out = out.astype(np.float32) / 255.0
        if layout == "chw":
            out = np.moveaxis(out, -1, 0)
        return out

    def _apply_agentview_occlusion(self, img):
        """
        Draw a can-like occluder over the red target cube in agentview.
        """
        hwc_u8, layout, dtype_kind = self._to_hwc_uint8(img)
        h, w, _ = hwc_u8.shape

        # crude red-object detector
        r = hwc_u8[..., 0].astype(np.int16)
        g = hwc_u8[..., 1].astype(np.int16)
        b = hwc_u8[..., 2].astype(np.int16)
        red_mask = (r > 90) & (r > g + 25) & (r > b + 25)

        if red_mask.any():
            ys, xs = np.where(red_mask)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
        else:
            cx, cy = w // 2, h // 2

        can_w = max(8, self.occluder_w)
        can_h = max(12, self.occluder_h)
        cap_h = max(6, can_h // 6)

        yy, xx = np.ogrid[:h, :w]

        body = (
            (np.abs(xx - cx) <= can_w // 2)
            & (yy >= cy - can_h // 2)
            & (yy <= cy + can_h // 2)
        )
        top = (((xx - cx) / (can_w / 2)) ** 2 + ((yy - (cy - can_h // 2)) / (cap_h / 2)) ** 2) <= 1
        bot = (((xx - cx) / (can_w / 2)) ** 2 + ((yy - (cy + can_h // 2)) / (cap_h / 2)) ** 2) <= 1

        # dark can body
        hwc_u8[body] = np.array([35, 35, 35], dtype=np.uint8)
        hwc_u8[top] = np.array([60, 60, 60], dtype=np.uint8)
        hwc_u8[bot] = np.array([20, 20, 20], dtype=np.uint8)

        # highlight strip
        highlight = (
            (xx >= cx - can_w // 4)
            & (xx <= cx - can_w // 4 + 2)
            & (yy >= cy - can_h // 2)
            & (yy <= cy + can_h // 2)
        )
        hwc_u8[highlight] = np.array([180, 180, 180], dtype=np.uint8)

        return self._from_hwc_uint8(hwc_u8, layout, dtype_kind)

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        obs = dict()
        for key in self.observation_space.keys():
            value = raw_obs[key]
            if self.occlude_agentview and key == "agentview_image":
                value = self._apply_agentview_occlusion(value)
            obs[key] = value

        # keep render cache aligned with policy input
        self.render_cache = obs[self.render_obs_key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            raw_obs = self.env.reset()

        obs = self.get_observation(raw_obs)
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img
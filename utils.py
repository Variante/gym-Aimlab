import math
import os
import random
from collections import deque

import numpy as np
import scipy.linalg as sp_la

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.util.shape import view_as_windows
from torch import distributions as pyd

import json
import win32gui
import time

def load_cfg():
    with open('./config.json', 'r', encoding="utf-8") as f:
        content = f.read()
        cfg = json.loads(content)
        return cfg


def get_possible_window_name(name="aim"):
    print("Search for the window whose name contains", name)
    possible_hwnd = None
    def winEnumHandler(hwnd, ctx):
        nonlocal possible_hwnd
        if win32gui.IsWindowVisible(hwnd):
            win_name = win32gui.GetWindowText(hwnd)
            if name in win_name:
                possible_hwnd = hwnd
    win32gui.EnumWindows(winEnumHandler, None)
    if possible_hwnd is None:
        print("Window not found")
    print('-' * 8)
    return possible_hwnd


def get_window_handle(name):
    handle = win32gui.FindWindow(0, name)
    # print(handle)
    # handle = 0xd0ea6
    if not handle:
        print("Can't not find " + name)
        handle = get_possible_window_name()
    return handle
    
def set_window_roi(name, target, padding):
    x1, y1, x2, y2 = (0, 0, 1, 1)
    ptop, pdown, pleft, pright = padding
    
    handle = get_window_handle(name)
    if handle is None:
        return {'top': -1, 'left': -1, 'width': 100, 'height': 100}
    win32gui.MoveWindow(handle, target[0], target[1], target[2], target[3], True)
    
    while win32gui.GetForegroundWindow()!= handle:
        time.sleep(1)
    
    
    window_rect = win32gui.GetWindowRect(handle)
    
    w = window_rect[2] - window_rect[0] - pleft - pright
    h = window_rect[3] - window_rect[1] - ptop - pdown
    
    window_dict = {
        'left': window_rect[0] + int(x1 * w) + pleft,
        'top': window_rect[1] + int(y1 * h) + ptop,
        'width': int((x2 - x1) * w),
        'height': int((y2 - y1) * h)
    }
    return window_dict
    
    
def get_window_roi(name, pos, padding):
    x1, y1, x2, y2 = pos
    ptop, pdown, pleft, pright = padding
    
    handle = get_window_handle(name)
    if handle is None:
        return {'top': -1, 'left': -1, 'width': 100, 'height': 100}
        
    window_rect = win32gui.GetWindowRect(handle)
    
    w = window_rect[2] - window_rect[0] - pleft - pright
    h = window_rect[3] - window_rect[1] - ptop - pdown
    
    window_dict = {
        'left': window_rect[0] + int(x1 * w) + pleft,
        'top': window_rect[1] + int(y1 * h) + ptop,
        'width': int((x2 - x1) * w),
        'height': int((y2 - y1) * h)
    }
    
    return window_dict


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


class FrameStack:
    def __init__(self, env, k):
        self._k = k
        self._frames = deque([], maxlen=k)
        self.env = env
        self.action_space = self.env.action_space
        
    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
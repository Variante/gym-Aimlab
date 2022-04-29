# -*- coding:utf-8 -*-

import win32gui
import win32api
import win32con
import win32com.client
import win32ui
import win32clipboard
import numpy as np
import cv2
import time
import random
import traceback
import mss
from utils import *
from ocr import *
from controller import *


class ActionSpace:
    def __init__(self, shape):
        self.shape = shape
    
    def sample(self):
        return np.random.rand(self.shape[0]) * 2 - 1

class AimlabGym:
    def __init__(self, cfg):
        self.control = VController()
        self.ocr = ImageOCR()
        self.cfg = cfg
        self.img_size = cfg['image_size']
        self.m = mss.mss()
        self.last_pts = 0
        
        self.action_space = ActionSpace((3, ))
        
    def seed(self, s):
        np.random.seed(s)
        
    def _fetch_img(self):
        cfg = self.cfg
        while True:
            win_info = set_window_roi(cfg['name'], cfg['target'], cfg['padding'])
            # print(win_info)
            if win_info['left'] < 0 and win_info['top'] < 0:
                time.sleep(0.1)
                continue
            break
        return np.array(self.m.grab(win_info))
        
    def return_obs(self, img):
        return np.moveaxis(cv.resize(slice_roi(img[:, :, :3], self.cfg['roi']), (self.img_size, self.img_size)), -1, 0)
        
    def step(self, action):
        cfg = self.cfg
        self.control.parse_action(action)
        time.sleep(cfg['action_last'])
        done = False
        while True:
            img = self._fetch_img()
            obs = self.return_obs(img)
            res = self.ocr.parse_img(img, cfg['data'])
            if res['pts'] >= 0:
                break
            done = self.ocr.check_done(img)
            self.control.reset()
            if done:
                break
            print("Failed to read score")
            time.sleep(1)        
        # print(res)
        reward = (res['pts'] - self.last_pts) / cfg['reward_scale'] - np.linalg.norm(action[:2]) * 0.1
        self.last_pts = res['pts']
        
        if done:
            reward = 0
            print("Done")
            # self.control.press_dpad_down()
        return obs, reward, done, res
        
    def reset(self):
        self.control.reset()
        self.control.press_dpad_down()
        while self.ocr.check_done(self._fetch_img()):
            self.control.press_enter()
            time.sleep(1)
        while not self.ocr.check_ready(self._fetch_img()):
            time.sleep(1)
        self.control.press_right_trigger()
        time.sleep(3)
        self.last_pts = 0
        img = self._fetch_img()
        return self.return_obs(img)

        
        
if __name__ == '__main__':
    cfg = load_cfg()
    env = AimlabGym(cfg)
    time.sleep(3)
    print("Start to train")
    while True:
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, _ = env.step(env.action_space.sample())
            
    cv2.destroyAllWindows()
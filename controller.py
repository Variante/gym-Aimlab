import vgamepad as vg
import numpy as np
import time
import win32com
import win32api

class VController:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        self.shell = win32com.client.Dispatch("WScript.Shell")

    def press_enter(self):
        self.shell.SendKeys("{ENTER}")
        win32api.Sleep(2500)

    def press(self, btn):
        self.gamepad.press_button(button=btn)
        self.gamepad.update()
        time.sleep(0.05)
        self.gamepad.release_button(button=btn)
        self.gamepad.update()
        
    def press_right_trigger(self, value=255, slp=0.05):
        self.gamepad.right_trigger(value=value)
        self.gamepad.update()
        time.sleep(slp)
        self.gamepad.right_trigger(value=0)
        self.gamepad.update()
        
    def press_dpad_down(self):
        self.press(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)

    def parse_action(self, action):
        action = np.clip(action, -1, 1)
        self.gamepad.right_trigger(value=255 if action[-1] > 0 else 0)
        self.gamepad.right_joystick_float(x_value_float=action[0], y_value_float=action[1])
        self.gamepad.update()
        time.sleep(0.05)
        self.gamepad.right_trigger(value=0) # release trigger
        self.gamepad.update()

    def reset(self):
        self.gamepad.reset()
        self.gamepad.update()

    def __del__(self, **argv):
        self.reset()
    

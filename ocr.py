# -*- coding:utf-8 -*-

import numpy as np
import cv2 as cv
from utils import *

def make_template():
    dst = np.hstack([unify_size(cv.imread(f'sample/{i}0.png', 0)) for i in range(10)])
    cv.imwrite('digits.png', dst)

def slice_xywh(img, roi):
    y, x = img.shape[:2]
    x1, x2 = roi[0], roi[2] + roi[0]
    y1, y2 = roi[1], roi[3] + roi[1]
    return img[y1:y2, x1:x2]
    
def unify_size(img, yx=(14, 12)):
    res = np.zeros(yx, dtype=np.uint8)
    res[:img.shape[0], :img.shape[1]] = img
    return res
    
def slice_roi(img, roi):
    y, x = img.shape[:2]
    x1, x2 = int(roi[0] * x), int(roi[2] * x)
    y1, y2 = int(roi[1] * y), int(roi[3] * y)
    return img[y1:y2, x1:x2]


class ImageOCR:
    def __init__(self):
        self.digit_template = cv.imread('digits.png', 0)
        self.ready_template = cv.imread('ready.png', 0)
        self.end_template = cv.imread('end.png', 0)
        
    def check_done(self, img):  
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(self.end_template, gray[:gray.shape[0]//2, :gray.shape[1]//2], cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(res)
        return max_val > 0.9

    def check_ready(self, img):  
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w = gray.shape[0], gray.shape[1]
        res = cv.matchTemplate(self.ready_template, gray[:h//4], cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(res)
        # print(max_val)
        return max_val > 0.9
    
    
    def recognize_digits(self, img, name=None):
        gui = img.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 250, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL  , cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return -1
        # filter contours
        bbox = [cv.boundingRect(i) for i in contours]
        filtered_combo = [(i, j) for i, j in zip(contours, bbox) if j[-1] > 10 and j[-1] < 20]
        if len(filtered_combo) == 0:
            return -1
        filtered_contours, filtered_bbox = (list(i) for i in zip(*filtered_combo))
        
        
        # print(filtered_contours)
        # cv.imshow(f'binary{name}', thresh)
        # [cv.imwrite(f'sample/t{i}.png', slice_xywh(thresh, j)) for i, j in enumerate(filtered_bbox)]
        # digits = np.vstack([unify_size(slice_xywh(thresh, i)) for i in filtered_bbox])
        # cv.imshow('digis', digits)
        # cv.drawContours(gui, filtered_contours, -1, (0,255,0), 3)
        # cv.imshow(f'g{name}', gui)
        
        def query_digits(src, dst):
            width = dst.shape[1] // 10
            if src.shape[0] > dst.shape[0]:
                tx = int(dst.shape[0] * src.shape[1] / src.shape[0])
                src = cv.resize(src, (tx, dst.shape[0]))
            res = cv.matchTemplate(src, dst, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(res)
            # print(max_loc)
            return max_loc[0] // width
        
        filtered_bbox.sort(key=lambda x: x[0])
        digits = [query_digits(slice_xywh(thresh, i), self.digit_template) for i in filtered_bbox]
        
        def combine_digits(d):
            res = 0
            for i in digits:
                if i < 0:
                    print("Parse digit error: ", d)
                    continue
                res *= 10
                res += i
            return res
            
        # cv.waitKey(0)    
        return combine_digits(digits)
        
    
    def parse_img(self, img, data):
        return {
            i['name']: self.recognize_digits(slice_roi(img, i['roi']), name=i['name'])
            for i in data    
        }

    
if __name__ == '__main__':
    
    cfg = load_cfg()
    img = cv.imread('96.png')
    ocr = ImageOCR()
    print(ocr.parse_img(img, cfg['data']))
    cv.imshow(f'preview', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    make_template()
    """

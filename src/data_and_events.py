import cv2 as cv
import os
from copy import copy
from enum import Enum

class data_and_events:
    drawing_status = Enum("drawing_status", ["NONE", "DRAWING", "DONE"])

    def __init__(self, image_name) -> None:
        self.start_point = (-1, -1)
        path = os.path.join(os.getcwd(), "..", "img", image_name)
        img = cv.imread(path)
        self.drawing = self.drawing_status.NONE
        self.img = copy(img)
        self.img_copy = copy(img)

    def mouse_callback(self, event, x, y, flags, param) -> None:
        if self.drawing == self.drawing_status.DONE:
            return

        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = self.drawing_status.DRAWING
            self.start_point = (x, y)
        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == self.drawing_status.DRAWING:
                self.img = copy(self.img_copy)
                cv.rectangle(self.img, self.start_point, (x, y), (0, 255, 0), 2)
        elif event == cv.EVENT_LBUTTONUP:
            self.drawing = self.drawing_status.NONE
            cv.rectangle(self.img, self.start_point, (x, y), (0, 255, 0), 2)

    def get_image(self):
        return self.img

    def keyboard_handler(self):
        key_press = cv.waitKey(1) & 0xFF
        if key_press == 27:  # esc key
            return -1
        elif key_press == ord("r"):  # reset image
            self.img = copy(self.img_copy)
            self.drawing = False
        elif key_press == 13:
            self.drawing = self.drawing_status.DONE
        return 0
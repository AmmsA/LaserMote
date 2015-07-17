from Point import Point
import matplotlib.pyplot as plt
import cv2
import sys
import math
import numpy as np


def imshow(img):
    # hide the x and y axis for images
    plt.axis('off')
    # RGB images are actually BGR in OpenCV, so convert before displaying
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # otherwise, assume it's gray scale and just display it
    else:
        plt.imshow(img, 'gray')

    plt.show()

    # Defined colors


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

class LaserMote(object):
    NOT_FOUND_TEXT = "Laser pointer is not detected."
    BOTTOM_LEFT_COORD = (25, 460)

    def __init__(self,
                 debug=False,
                 min_hue=25, max_hue=179,
                 min_sat=100, max_sat=255,
                 min_val=200, max_val=255,
                 min_area=2, max_area=300,
                 reset_time=None, wait_time=5,
                 distance_threshold=25,
                 tracking_method=1):

        """
        :param min_hue: minimum hue allowed
        :param max_hue: maximum hue allowed
        :param min_sat: minimum saturation allowed
        :param max_sat: maximum saturation allowed
        :param min_val: minimum value allowed
        :param max_val: maximum value allowed
        :param min_area: minimum area of the laser dot to look for
        :param max_area: maximum area of the laser dot to look for
        :param reset_time: time threshold to reset the last seen laser dot if not seen
        :param wait_time: the wait time to execute an action "turn on tv, print something, etc"
        :param distance_threshold: threshold of the distance between current point location and the last seen point location
        :param tracking_method: which tracking method to use
        :return:
        """

        self.min_hue = min_hue
        self.max_hue = max_hue
        self.min_sat = min_sat
        self.max_sat = max_sat
        self.min_val = min_val
        self.max_val = max_val

        self.min_area = min_area
        self.max_area = max_area

        if not reset_time:
            if tracking_method == 1:
                self.reset_time = 1.5
            elif tracking_method == 2:
                self.reset_time = 2
        else:
            self.reset_time = 1

        self.wait_time = wait_time

        self.camera = None

        self.debug = debug

        self.background = None
        self.blacked = None

        self.point = Point(reset_time=self.reset_time, wait_time=self.wait_time, debug=self.debug)
        self.point.setName('PointThread')
        self.point.daemon = True  # make deamon thread to terminate when main program ends
        self.point.start()

        self.distance_threshold = distance_threshold

        self.tracking_method = tracking_method

    # computes distance between last_seen point (x, y) and the input's (x,y)
    def get_distance(self, x2, y2):
        x1, y1 = self.point.get_last_seen_coordinates()
        dist = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
        return dist

    def setup_capture(self):
        # capture at location 0
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            sys.stderr.write("Error capturing camera at location 0. Quitting.\n")
            sys.exit(1)

        return self.camera

    def debug_text(self, cx, cy, area):
        return "Laser point detected at ({cx},{cy}) with area {area}.".format(cx=cx, cy=cy, area=area)

    # tracking method 1
    def in_laser_color_range(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Normal masking algorithm
        lower_red = np.array([self.min_hue, self.min_sat, self.min_val])
        upper_red = np.array([self.max_hue, self.max_sat, self.max_val])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        laser = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(laser, cv2.COLOR_BGR2GRAY)

        return hsv, gray

    # tracking method 2
    def get_hsv(self, frame):
        hsv = cv2.cvtColor(frame, cv2.cv.CV_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        (t, tmp) = cv2.threshold(hue, self.max_hue, 0, cv2.THRESH_TOZERO_INV)
        (t, hue) = cv2.threshold(tmp, self.min_hue, 255, cv2.THRESH_BINARY, hue)
        hue = cv2.bitwise_not(hue)

        (t, tmp) = cv2.threshold(sat, self.max_sat, 0, cv2.THRESH_TOZERO_INV)
        (t, sat) = cv2.threshold(tmp, self.min_sat, 255, cv2.THRESH_BINARY, sat)

        (t, tmp) = cv2.threshold(val, self.max_sat, 0, cv2.THRESH_TOZERO_INV)
        (t, val) = cv2.threshold(tmp, self.min_val, 255, cv2.THRESH_BINARY, val)

        if self.debug:
            cv2.imshow('sat', sat)
            cv2.imshow('hue', hue)
            cv2.imshow('val', val)

        laser = cv2.bitwise_and(hue, val)
        laser = cv2.bitwise_and(sat, laser, laser)

        hsv = cv2.merge([hue, sat, val])

        return hsv, laser

    def display(self, laser, frame):
        contours, hierarchy = cv2.findContours(
            laser, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        found_valid_point = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            # check if the contour is withing the area threshold
            if self.min_area < area < self.max_area:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # if self.debug:
                #     print "within threshold " + str(area)

                if self.point.was_seen() \
                        and self.get_distance(cx, cy) <= self.distance_threshold:

                    if self.debug:
                        print '[DEBUG] Distance between last seen point (' + str(self.point.last_seen_x) + "," + str(
                            self.point.last_seen_y) + ") and current point (" + str(cx) + "," + str(cy) + ") :" + str(
                            self.get_distance(cx, cy))

                    # update last point locations
                    self.point.update_last_seen_position(cx, cy)
                    self.point.set_on()

                    # cv2.drawContours(frame, cnt, -1, GREEN, 5) # draws the contour
                    cv2.circle(frame, (cx, cy), 1, GREEN, thickness=4, lineType=8, shift=0)  # circle cnt center

                    cv2.putText(
                        frame,
                        self.debug_text(cx, cy, area), LaserMote.BOTTOM_LEFT_COORD,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                elif not self.point.was_seen():
                    self.point.set_on()
                    self.point.update_last_seen_position(cx, cy)
                    if self.debug:
                        print '[DEBUG] First seen point is at: (' + str(cx) + "," + str(cy) + ")."
                        print '[DEBUG] Updated last seen coordinates.'

                found_valid_point = True
                break  # only one contour

        if not found_valid_point:
            cv2.putText(frame, LaserMote.NOT_FOUND_TEXT,
                        LaserMote.BOTTOM_LEFT_COORD, cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

            if self.point.is_on() and self.point.was_seen():
                self.point.set_off()

        return frame

    def run(self):

        self.setup_capture()
        while True:
            # get frame
            ret, frame = self.camera.read()

            # get frame size
            w, h = frame.shape[:2]

            # mirror the frame
            # frame = cv2.flip(frame, 1)

            mask = None
            # wait for space to be clicked to capture background
            if cv2.waitKey(1) == 32:
                print 'background captured'
                self.background = frame
                # clean up noise by adding median blur
                cv2.medianBlur(self.background, 5, self.background)

                self.blacked = cv2.bitwise_and(frame, frame, mask=mask)

            if self.debug and not (self.blacked is None):
                # absDiff = cv2.absdiff(self.background, frame)
                # print absDiff
                # kernel = np.ones((1, 1), np.uint8)
                # mask = cv2.cvtColor(absDiff, cv2.COLOR_BGR2GRAY, mask)
                #
                # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, mask, iterations=5)
                #
                # ret, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV, mask)
                #
                # mask = cv2.bitwise_not(mask, mask)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_background = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
                diff_image = cv2.absdiff(gray, gray_background)
                cv2.threshold(diff_image, 20, 255, cv2.THRESH_BINARY, diff_image)

                element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                cv2.erode(diff_image, element, diff_image, iterations=1)
                cv2.dilate(diff_image, element, diff_image, iterations=2)

                self.blacked = cv2.bitwise_and(frame, frame, mask=diff_image)
                cv2.imshow('blacked', self.blacked)
                cv2.imshow('mask', diff_image)

            if not (self.blacked is None):
                hsv, laser = self.get_hsv(self.blacked)
            elif self.tracking_method == 2:
                hsv, laser = self.get_hsv(frame)
            elif self.tracking_method == 1:
                hsv, laser = self.in_laser_color_range(frame)

            res = self.display(laser, frame)

            if self.debug:
                cv2.imshow('laser', laser)
                cv2.imshow('hsv', hsv)

            cv2.imshow('result', res)

            # wait for space to save single frame
            # if cv2.waitKey(5) == 32:
            #     singleFrame = frame
            #     imshow(singleFrame)

            # exit on ESC press
            if cv2.waitKey(5) == 27:
                # clean up
                cv2.destroyAllWindows()
                self.camera.release()
                break


if __name__ == '__main__':
    LaserMote = LaserMote(min_hue=154, min_sat=40, min_val=200, debug=True, tracking_method=1, wait_time=5)
    LaserMote.run()

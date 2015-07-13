import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys


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


class LaserMote(object):
    def __init__(self,
                 debug = False,
                 min_hue=25, max_hue=180,
                 min_sat=100, max_sat=255,
                 min_val=170, max_val=255):

        """
        :param min_hue: minimum hue allowed
        :param max_hea: maximum hue allowed
        :param min_sat: minimum saturation allowed
        :param max_sat: maximum saturation allowed
        :param min_val: minimum value allowed
        :param max_val: maximum value allowed
        :param width:
        :param height:
        :return:
        """

        self.min_hue = min_hue
        self.max_hue = max_hue
        self.min_sat = min_sat
        self.max_sat = max_sat
        self.min_val = min_val
        self.max_val = max_val

        self.camera = None

        self.debug = debug

        self.background = None
        self.blacked = None

    def setup_capture(self):
        # capture at location 0
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            sys.stderr.write("Error capturing camera at location 0. Quitting.\n")
            sys.exit(1)

        return self.camera

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

        for cnt in contours:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if (area > 5 and area < 100):
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.drawContours(frame, cnt, -1, (0, 0, 255), 5)

                cv2.putText(
                    frame, str(area), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if self.debug:
                    pass
                    #print "within threshold " + str(area)

        return frame

    def run(self):

        self.setup_capture()
        while True:
            # get frame
            ret, frame = self.camera.read()

            # get frame size
            w, h = frame.shape[:2]

            # mirror the frame
            frame = cv2.flip(frame, 1)

            mask = None
            # wait for space to be clicked to capture background
            if cv2.waitKey(1) == 32:
                print 'background captured'
                self.background = frame
                # clean up noise by adding median blur
                cv2.medianBlur(self.background,5, self.background)

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
                grayb = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
                diff_image = cv2.absdiff(gray, grayb)
                cv2.threshold(diff_image, 20, 255, cv2.THRESH_BINARY, diff_image)

                element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                cv2.erode(diff_image, element, diff_image, iterations=1)
                cv2.dilate(diff_image, element, diff_image, iterations=2)

                self.blacked = cv2.bitwise_and(frame, frame, mask=diff_image)
                cv2.imshow('blacked', self.blacked)
                cv2.imshow('mask', diff_image)

            if not (self.blacked is None):
                hsv, laser = self.get_hsv(self.blacked)
            else:
                hsv, laser = self.get_hsv(frame)

            res = self.display(laser, frame)

            if self.debug:
                cv2.imshow('laser', laser)
                cv2.imshow('hsv', hsv)

            cv2.imshow('result', res)

            # wait for space to save single frame
            if cv2.waitKey(5) == 32:
                singleFrame = frame
                # imshow(singleFrame)

            # exit on ESC press
            if cv2.waitKey(5) == 27:
                # clean up
                cv2.destroyAllWindows()
                self.camera.release()
                break


if __name__ == '__main__':
    LaserMote = LaserMote(min_hue=20, max_hue=160, debug=True)
    LaserMote.run()

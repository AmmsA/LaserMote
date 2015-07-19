__author__ = "Mustafa S"

from ObjectLocation import ObjectLocation
from Point import Point
import matplotlib.pyplot as plt
import cv2
import argparse
import sys
import math
import numpy as np
import time


def image_show(img):
    """
    Shows the image in matplotlib window.

    :param img: image to show.
    :rtype : void
    """
    plt.axis('off')
    # RGB images are actually BGR in OpenCV, so convert before displaying.
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # otherwise, assume it's gray scale and just display it.
    else:
        plt.imshow(img, 'gray')

    plt.show()


# Defined colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class LaserMote(object):
    NOT_FOUND_TEXT = "Laser pointer is not detected."
    BOTTOM_LEFT_COORD = (25, 460)

    def __init__(self,
                 min_hue=155, max_hue=179,
                 min_sat=100, max_sat=255,
                 min_val=200, max_val=255,
                 min_area=2, max_area=300,
                 reset_time=None, wait_time=5,
                 distance_threshold=25,
                 tracking_method=1,
                 capture_locations_flag=True,
                 locations_size=1,
                 write_to_video=False,
                 debug=False, ):

        """
        Initializes all needed variables.

        :param capture_locations_flag: __________________________
        :param min_hue: minimum hue allowed.
        :param max_hue: maximum hue allowed.
        :param min_sat: minimum saturation allowed.
        :param max_sat: maximum saturation allowed.
        :param min_val: minimum value allowed.
        :param max_val: maximum value allowed.
        :param min_area: minimum area of the laser dot to look for.
        :param max_area: maximum area of the laser dot to look for.
        :param reset_time: time threshold to reset the last seen laser dot if not seen.
        :param wait_time: the wait time to execute an action "turn on tv, print something, etc".
        :param distance_threshold: threshold of the distance between current point location and last seen point location.
        :param tracking_method: which tracking method to use.
        :param write_to_video:
        :param debug: boolean to allow/disallow debug printing and extra windows.
        :rtype : LaserMote object.
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
        self.out = None
        self.write_to_video = write_to_video

        self.capture_locations_flag = capture_locations_flag
        if not capture_locations_flag:
            self.locations_size = 0
        else:
            self.locations_size = locations_size

        cv2.namedWindow('result')
        cv2.setMouseCallback("result", self.capture_locations, 0)

        self.debug = debug

        self.result = None
        self.background = None
        self.blacked = None

        self.point = Point(reset_time=self.reset_time, wait_time=self.wait_time, debug=self.debug)
        self.point.setName('PointThread')
        self.point.daemon = True  # make deamon thread to terminate when main program ends
        self.point.start()

        self.distance_threshold = distance_threshold

        self.tracking_method = tracking_method

        self.reference_points = []
        self.object_locations = []

    def capture_locations(self, event, x, y, flags, params):
        """
        Captures mouse event for creating the ROIs

        :param event:
        :param x:
        :param y:
        :param flags:
        :param params:
        """
        for i in xrange(self.locations_size):
            reference_points = self.reference_points

            # record the (x,y) coordinates when the mouse is clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                reference_points.append((x, y))
                print len(reference_points)

            # when left mouse button is released
            elif event == cv2.EVENT_LBUTTONUP:
                size = len(reference_points)
                if size > 0:
                    # cv2.rectangle(self.result, reference_points[size - 2], (x, y), (0, 255, 0), 2)
                    o = ObjectLocation(reference_points[size - 1][0],
                                       reference_points[size - 1][1],
                                       x,
                                       y)
                    self.object_locations.append(o)

    def show_object_locations(self):
        """
        Draws a rectangle around a user specified ROI.
        Asks the user to enter a name for that ROI.
        ROI must be provided with a name, otherwise it will be disregarded.

        """
        locations = self.object_locations

        for o in locations:
            if o.name is None:
                print 'Enter name of ROI: ',
                o.name = sys.stdin.readline().strip()

                # name must be supplied
                if len(o.name) < 1:
                    print "You haven't supplied a name for this ROI. Please try again"
                    self.object_locations.remove(o)
                    continue
            cv2.rectangle(self.result, (o.x1, o.y1), (o.x2, o.y2), (0, 255, 0), 1)
            cv2.putText(
                self.result,
                o.name, (o.x1, o.y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)

            # coordinates printing
            # cv2.putText(
            #     self.result,
            #     str((o.x1, o.y1)), (o.x1, o.y1),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            #
            # cv2.putText(
            #     self.result,
            #     str((o.x2, o.y1)), (o.x2, o.y1),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            #
            # cv2.putText(
            #     self.result,
            #     str((o.x2, o.y2)), (o.x2, o.y2),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            #
            # cv2.putText(
            #     self.result,
            #     str((o.x1, o.y2)), (o.x1, o.y2),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    def is_dot_within_rois(self, x, y):
        """

        :param x:
        :param y:
        :return: ObjectLocation object
        """
        for o in self.object_locations:
            bottom_left = (o.x1, o.y1)
            bottom_right = (o.x2, o.y1)
            top_right = (o.x2, o.y2)
            top_left = (o.x1, o.y2)

            if x < bottom_left[0]:
                continue
            if x > bottom_right[0]:
                continue
            if y < top_right[1]:
                continue
            if y > bottom_right[1]:
                continue

            return o

        return None

    def get_distance(self, x2, y2):
        """
        Calculates the Euclidean distance between the given point (x2,y2) and last seen point.

        :param x2: x axis value.
        :param y2: y axis value.
        :return: the Euclidean distance between the given point (x2,y2) and last seen point.
        :rtype : double
        """
        x1, y1 = self.point.get_last_seen_coordinates()
        dist = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
        return dist

    def setup_video(self):
        # create VideoWriter object
        fps = 12
        self.out = cv2.VideoWriter('RealWorld'
                                   '.avi', -1, fps, (640, 480), True)
        print self.out
    def setup_capture(self):
        """
        Setups the camera capture.

        :return: camera capture.
        :rtype: VideoCapture object
        """
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            sys.stderr.write("Error capturing camera at location 0. Quitting.\n")
            sys.exit(1)

        return self.camera

    @staticmethod
    def time_in_h_m_s():
        return time.strftime('%H:%M:%S')

    def debug_text(self, cx=None, cy=None, area=None, found=False):
        """
        Returns text describing location and area of laser dot, if found.
        Otherwise return not found debug text

        :param cx: x axis value.
        :param cy: y axis value.
        :param area: approximation of laser dot area.
        :param found: is dot found or not
        :rtype : String
        """
        if found:
            return "{time} Laser point detected at ({cx},{cy}) with area {area}." \
                .format(time=self.time_in_h_m_s(), cx=cx, cy=cy, area=area)
        else:
            return "{time} {text}".format(time=self.time_in_h_m_s(), text=LaserMote.NOT_FOUND_TEXT)

    def in_laser_color_range(self, frame):
        """
        Tracking method 1.

        :param frame: frame to threshold.
        :return: hsv image, and image within our threshold.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Normal masking algorithm
        lower_red = np.array([self.min_hue, self.min_sat, self.min_val])
        upper_red = np.array([self.max_hue, self.max_sat, self.max_val])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        laser = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(laser, cv2.COLOR_BGR2GRAY)

        return hsv, gray

    def get_hsv(self, frame):
        """
        Tracking method 2.

        :param frame: frame to threshold it's hsv values.
        :return: hsv image, and image withing the threshold.
        """
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
        """
        Finds contour and determines if it's valid laser dot or not through area threshold.
        Displays debug text and draws a circle on detected laser point.

        :param laser: image containing only laser dot (i.e image within our threshold).
        :param frame: frame to display and writing debug text on.
        :return: frame.
        """
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
                        :  # and self.get_distance(cx, cy) <= self.distance_threshold:

                    if self.debug:
                        print "[DEBUG] Distance between last seen point ({0},{1})" \
                              " and current point ({2},{3}) :{4}".format(str(self.point.last_seen_x),
                                                                         str(self.point.last_seen_y),
                                                                         str(cx), str(cy),
                                                                         str(self.get_distance(cx, cy)))

                    # update last point locations
                    self.point.update_last_seen_position(cx, cy)
                    self.point.set_on()
                    self.point.current_object = self.is_dot_within_rois(cx, cy)

                    #cv2.drawContours(frame, cnt, -1, GREEN, 15)  # draws the contour
                    cv2.circle(frame, (cx, cy), 1, GREEN, thickness=4, lineType=8, shift=0)  # circle cnt center

                elif not self.point.was_seen():
                    #cv2.drawContours(frame, cnt, -1, BLUE, 15)  # draws the contour
                    self.point.set_on()
                    self.point.update_last_seen_position(cx, cy)
                    self.point.current_object = self.is_dot_within_rois(cx, cy)

                    if self.debug:
                        print '[DEBUG] First seen point is at: ({0},{1}).'.format(str(cx), str(cy))
                        print '[DEBUG] Updated last seen coordinates.'

                cv2.putText(
                    frame,
                    self.debug_text(cx, cy, area, found=True), LaserMote.BOTTOM_LEFT_COORD,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                found_valid_point = True
                break  # only one contour

        if not found_valid_point:
            cv2.putText(frame, self.debug_text(),
                        LaserMote.BOTTOM_LEFT_COORD, cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

            if self.point.is_on() and self.point.was_seen():
                current_object = self.is_dot_within_rois(self.point.last_seen_x, self.point.last_seen_y)
                self.point.set_off(current_object)

        return frame

    def run(self):

        self.setup_capture()
        if self.write_to_video:
            self.setup_video()
        while True:
            # get frame
            ret, frame = self.camera.read()

            # get frame size
            # w, h = frame.shape[:2]

            # mirror the frame
            frame = cv2.flip(frame, 1)

            mask = None
            # wait for space to be clicked to capture background
            if cv2.waitKey(1) == 32:
                print 'background captured'
                self.background = frame
                # clean up noise by adding median blur
                cv2.medianBlur(self.background, 5, self.background)

                self.blacked = cv2.bitwise_and(frame, frame, mask=mask)

            if self.debug and (self.blacked is not None):
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

            self.result = self.display(laser, frame)

            if len(self.object_locations) > 0:
                self.show_object_locations()

            cv2.imshow('result', self.result)

            if self.write_to_video:
                self.out.write(self.result)  # writes to video

            if self.debug:
                cv2.imshow('laser', laser)
                cv2.imshow('hsv', hsv)

            # wait for space to save single frame
            # if cv2.waitKey(5) == 32:
            #     singleFrame = frame
            #     image_show(singleFrame)

            # exit on ESC press
            if cv2.waitKey(5) == 27:
                # clean up
                cv2.destroyAllWindows()
                self.camera.release()
                if self.write_to_video:
                    self.out.release()
                    self.out = None
                break


'''
:param capture_locations_flag: __________________________
        :param min_hue: minimum hue allowed.
        :param max_hue: maximum hue allowed.
        :param min_sat: minimum saturation allowed.
        :param max_sat: maximum saturation allowed.
        :param min_val: minimum value allowed.
        :param max_val: maximum value allowed.
        :param min_area: minimum area of the laser dot to look for.
        :param max_area: maximum area of the laser dot to look for.
        :param reset_time: time threshold to reset the last seen laser dot if not seen.
        :param wait_time: the wait time to execute an action "turn on tv, print something, etc".
        :param distance_threshold: threshold of the distance between current point location and last seen point location.
        :param tracking_method: which tracking method to use.
        :param write_to_video:
        :param debug: boolean to allow/disallow debug printing and extra windows.
        :rtype : LaserMote object.

        min_hue=25, max_hue=179,
                 min_sat=100, max_sat=255,
                 min_val=200, max_val=255,
                 min_area=2, max_area=300,
                 reset_time=None, wait_time=5,
                 distance_threshold=25,
                 tracking_method=1,
                 capture_locations_flag=True,
                 locations_size=1,
                 write_to_video=False,
                 debug=False
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LaserMote')
    parser.add_argument('-m', '--min_hue',
                        default=165,
                        type=int,
                        help='Minimum Hue'
                        )
    parser.add_argument('-M', '--max_hue',
                        default=179,
                        type=int,
                        help='Maximum Hue'
                        )
    parser.add_argument('-s', '--min_sat',
                        default=40,
                        type=int,
                        help='Minimum Saturation'
                        )
    parser.add_argument('-S', '--max_sat',
                        default=255,
                        type=int,
                        help='Maximum Saturation'
                        )
    parser.add_argument('-v', '--min_val',
                        default=200,
                        type=int,
                        help='Minimum Value'
                        )
    parser.add_argument('-V', '--max_val',
                        default=255,
                        type=int,
                        help='Maximum Value'
                        )
    parser.add_argument('-a', '--min_area',
                        default=2,
                        type=int,
                        help='Minimum area of dot'
                        )
    parser.add_argument('-A', '--max_area',
                        default=300,
                        type=int,
                        help='Maximum area of dot'
                        )
    parser.add_argument('-r', '--reset_time',
                        default=0,
                        type=int,
                        help='Time threshold to reset the last seen laser dot if not seen.'
                        )
    parser.add_argument('-w', '--wait_time',
                        default=5,
                        type=int,
                        help='The wait time to execute an action "turn on tv, print something, etc".'
                        )
    parser.add_argument('-d', '--distance_threshold',
                        default=25,
                        type=int,
                        help='Threshold of the distance between current point location and last seen point location.'
                        )
    parser.add_argument('-t', '--tracking_method',
                        default=1,
                        type=int,
                        help='Which tracking method to use <1,2>?'
                        )
    parser.add_argument('-c', '--capture_locations_flags',
                        default=True,
                        type=bool,
                        help='Are we allowed to set ROIs?'
                        )
    parser.add_argument('-l', '--locations_size',
                        default=1,
                        type=int,
                        help='Number of ROIs to be set'
                        )
    parser.add_argument('-R', '--write_to_video',
                        default=False,
                        type=bool,
                        help='Set video recording on/off'
                        )
    parser.add_argument('-debug', '--debug',
                        default=False,
                        type=bool,
                        help='Show debug messages and windows'
                        )
    params = parser.parse_args()


    # LaserMote = LaserMote(min_hue=154, min_sat=40, min_val=200, debug=False, tracking_method=1, wait_time=5,
    #                       write_to_video=True)

    LaserMote = LaserMote(min_hue=params.min_hue,
                          max_hue=params.max_hue,
                          min_sat=params.min_sat,
                          max_sat=params.max_sat,
                          min_val=params.min_val,
                          max_val=params.max_val,
                          min_area=params.min_area,
                          max_area=params.max_area,
                          reset_time=params.reset_time,
                          wait_time=params.wait_time,
                          distance_threshold=params.distance_threshold,
                          tracking_method=params.tracking_method,
                          capture_locations_flag=params.capture_locations_flags,
                          locations_size=params.locations_size,
                          write_to_video=params.write_to_video,
                          debug=params.write_to_video)
    LaserMote.run()

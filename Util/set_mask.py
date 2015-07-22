#!/usr/bin/env python

import argparse
import os

import cv2
import numpy as np

from tracker.analysis import *
from tracker.command import *
from tracker.data import *
from tracker.utils import *

curr_point = None
poly_points = []
poly_changed = False


#=========================================================================
# Call back function
#=========================================================================

def on_mouse(e, mx, my, lb, rb):
    global curr_point
    global poly_points
    global poly_changed

    if lb == 1:
        if curr_point is None:
            d_min = 10000
            for (i, (x, y)) in enumerate(poly_points[0]):
                d = np.sqrt((mx - x)**2 + (my - y)**2)
                if(d < d_min):
                    d_min = d
                    curr_point = i
        else:
            poly_points[0][curr_point][0] = mx
            poly_points[0][curr_point][1] = my
            poly_changed = True
    elif lb == 2:
        poly_points = np.array(
            np.concatenate((poly_points, [[[mx, my]]]), 1), dtype=np.int32)
        poly_changed = True
    else:
        curr_point = None


#=========================================================================
# Main function
#=========================================================================

def main(prj, delay, jump):
    global curr_point
    global poly_points
    global poly_changed

    lum_thresh = prj.thresh[0]

    # start video capture
    capture = cv2.VideoCapture(prj.video_file)

    frame_height = capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    frame_width = capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

    # get the poly from the project
    poly_points = [np.array(prj.poly, dtype=np.int32)]

    (x, y, side, thresh, greater) = prj.laser
    laser_points = [np.array(
        [(x, y), (x + side, y), (x + side, y + side), (x, y + side)], dtype=np.int32)]

    # jump to the specified frame
    if jump > 0:
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, jump)

    first = True
    while True:
        # get the frame from the video
        (ret, frame) = capture.read()

        if not ret:
            break

        # split the color channels
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        [hue, sat, lum] = cv2.split(hsv)

        # make the mask
        mask = make_mask(poly_points, frame_height, frame_width)

        # play with the luminosity to get the black mouse
        (ret, lum_bin) = cv2.threshold(
            lum, lum_thresh, 255, cv2.THRESH_BINARY_INV)
        lum_bin = np.bitwise_and(lum_bin, mask)
        lum_bin = cv2.erode(lum_bin, kernel=cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)), iterations=2)

        (laser_on, laser_value) = detect_laser(frame, prj.laser)

        if laser_on:
            laser_color = (0, 0, 255)
        else:
            laser_color = (0, 0, 0)

        cv2.polylines(frame, laser_points, True, laser_color, 2)

        cv2.polylines(frame, poly_points, True, (0, 0, 255), 1)

        for (i, (x, y)) in enumerate(poly_points[0]):
            if(curr_point == i):
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            cv2.circle(frame, (x, y), 4, color, -1)

        cv2.putText(frame, "Laser square mean: %d" % laser_value, (10, int(
            frame_height) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
        cv2.putText(frame, "press 's' (save), 'q' (quit)", (10, int(
            frame_height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))

        cv2.imshow("Original", frame)
        cv2.imshow("Mouse", lum_bin)

        if first:
            first = False
            cv2.setMouseCallback("Original", on_mouse, 0)

        key = cv2.waitKey(delay)

        if key in (113, 115):
            break

        if not ret:
            key = raw_input()

    if poly_changed and (key == 115):
        prj.poly = poly_points[0].tolist()
        prj.save()


# Mandatory function to work with 'generic_gui' module. Should return a
# fully parametrized 'Command' object.
def cfg():
    command = Command("Mask")
    command.add_arg(Argument(name="Project File",  atype="dir",
                             cmd_name="project", desc="project directory name"))
    command.add_arg(Argument(name="Frame delay",   atype="int", cmd_name="--delay", cmd_option="-d",
                             default=1, minimum=0, maximum=100000,    desc="set the delay between frames (ms)."))
    command.add_arg(Argument(name="Jump to frame", atype="int", cmd_name="--jump",
                             cmd_option="-j", default=0, minimum=0, maximum=100000,    desc="jump to a specific frame."))

    return command

# Mandatory function to work with 'generic_gui' module. Receives the
# arguments as they were entered on the gui or in the command line


def run(args, io):
    global delay

    # open the project
    check_dir(io, args.project, "Project File")
    prj = Project(args.project)

    main(prj, args.delay, args.jump)

    cv2.destroyAllWindows()

#
# If the script is called directly from the command line
#
if __name__ == "__main__":
    # get the arguments from the command line
    args = cfg().get_parser().parse_args()
    run(args, InOutTerminal())

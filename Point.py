__author__ = 'Mustafa'
from threading import Thread
import time


class Point(Thread):
    def __init__(self, reset_time=2,
                 wait_time=5,
                 seen=False,
                 debug=False):

        Thread.__init__(self)

        self.reset_time = reset_time

        self.found_time = None  # first found time
        self.last_found_time = None  # the end time (when it was last disapeared)
        self.wait_time = wait_time
        print 'start time:', self.reset_time

        self.seen = seen
        self.last_seen_x = None
        self.last_seen_y = None

        self.on = False

        self.debug = debug

    def run(self):
        while True:
            if self.last_found_time:
                elapsed = (time.time() - self.last_found_time)
                if elapsed >= self.reset_time and self.seen and self.is_off():
                    self.seen = False
                    self.found_time = None
                    if self.debug:
                        print '[DEBUG] Resetting seen to false. (automatic reset after #' + str(
                            self.reset_time) + " seconds)"

    def update_last_seen_position(self, x, y):
        self.last_seen_x = x
        self.last_seen_y = y

    def get_last_seen_coordinates(self):
        return self.last_seen_x, self.last_seen_y

    def set_on(self):
        if not self.seen:
            self.found_time = time.time()
            self.seen = True
            print 'seen is now', self.seen
        self.on = True

    def set_off(self):
        self.on = False
        self.last_found_time = time.time()
        if self.found_time:
            elapsed = (time.time() - self.found_time)
            if elapsed >= self.wait_time:
                print '[DEBUG] Gesture detected. (ON for ' + str(self.wait_time) + " seconds)"

    def was_seen(self):
        return self.seen

    def is_on(self):
        return self.on

    def is_off(self):
        return not self.on

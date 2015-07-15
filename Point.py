__author__ = 'Mustafa'
from threading import Thread
import time

class Point(Thread):
    def __init__(self, wait_time=2,
                 seen=False,
                 debug=False):

        Thread.__init__(self)

        self.wait_time = wait_time
        self.seen = seen

        self.last_seen_x = None
        self.last_seen_y = None

        self.start_time = time.time()
        print 'start time:', self.start_time

        self.debug = debug

    def run(self):
        while True:
            elapsed = (time.time() - self.start_time)
            if elapsed >= self.wait_time and self.seen:
                self.seen = False
                if self.debug:
                    print '[DEBUG] Resetting seen to false. (automatic reset after #' + str(self.wait_time) + " seconds)"

    def update_last_seen_position(self, x, y):
        self.last_seen_x = x
        self.last_seen_y = y

    def get_last_seen_coordinates(self):
        return self.last_seen_x, self.last_seen_y

    def set_on(self):
        self.start_time = time.time()  # reset time
        self.seen = True

    def was_seen(self):
        return self.seen

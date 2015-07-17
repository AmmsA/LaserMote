__author__ = "Mustafa S"
from threading import Thread
import time


class Point(Thread):
    def __init__(self, reset_time=2,
                 wait_time=5,
                 seen=False,
                 debug=False):

        """
        Initializes all variables

        :param reset_time: how long should we wait after we last see the laser dot to  reset laser dot 'seen' to false
        :param wait_time: how long should we wait to determine if a gesture is detected
        :param seen: boolean to check if we saw the laser dot 'reset_time' seconds ago
        :param debug: boolean to determine whether to print debug messages or not
        """
        Thread.__init__(self)

        self.reset_time = reset_time

        self.found_time = None  # first found time
        self.last_found_time = None  # the end time (when it was last disappeared)
        self.wait_time = wait_time

        self.seen = seen
        self.last_seen_x = None
        self.last_seen_y = None

        self.on = False

        self.debug = debug

    def run(self):
        """
        Method to be run in a different thread. Checks if it has been over 'reset_time' since the last
        time we found laser dot.

        :rtype: void

        """
        while True:
            if self.last_found_time is not None:
                elapsed = (time.time() - self.last_found_time)
                if elapsed >= self.reset_time and self.seen and self.is_off:
                    self.seen = False
                    self.found_time = None
                    if self.debug:
                        print "[DEBUG] Resetting seen to false. (automatic reset after #{0} seconds)".format(str(
                            self.reset_time))

    def update_last_seen_position(self, x, y):
        """
        Updates the last seen coordinates of the laser dot

        :param x: x axis value
        :param y: y axis value
        :rtype: void
        """
        self.last_seen_x = x
        self.last_seen_y = y

    @property
    def get_last_seen_coordinates(self):
        """
        Returns the coordinates of where we last saw the laser dot

        :return: coordinates of last seen dot
        :rtype : (int,int)

        """
        return self.last_seen_x, self.last_seen_y

    def set_on(self):
        """
        Sets 'on' to true (i.e our laser is turned detected)

        :rtype : void
        """
        if not self.seen:
            self.found_time = time.time()
            self.seen = True
            if self.debug:
                print "[DEBUG] seen is now {0}".format(str(self.seen))
        self.on = True

    def set_off(self):
        """
        Sets 'on' to false (i.e our laser is turned not detected)

        :rtype : void
        """
        self.on = False
        self.last_found_time = time.time()
        if self.found_time is not None:
            elapsed = (time.time() - self.found_time)
            if elapsed >= self.wait_time:
                print "[DEBUG] Gesture detected. (ON for {0} seconds)".format(str(self.wait_time))

    @property
    def was_seen(self):
        """
        Returns true if laser dot was seen before it was rested (after 'reset_time' seconds)

        :return: seen or not
        :rtype: boolean
        """
        return self.seen

    @property
    def is_on(self):
        """
        Returns true if laser dot is currently on

        :return: on or not?
        :rtype: boolean
        """
        return self.on

    @property
    def is_off(self):
        """
        Returns true if laser dot is currently off

        :return: off or not?
        :rtype: boolean
        """
        return not self.on

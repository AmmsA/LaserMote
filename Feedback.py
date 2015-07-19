__author__ = 'Mustafa'

from threading import Thread
import pyttsx


class Feedback(Thread):
    def __init__(self, command=None):
        Thread.__init__(self)

        self.command = command
        self.engine = pyttsx.init()

    def run(self):
        self.speak()

    def speak(self):
        if self.command is not None:
            rate = self.engine.getProperty('rate')
            self.engine.setProperty('rate', rate - 50)  # changing speech rate
            self.engine.say(self.command)
            self.engine.runAndWait()

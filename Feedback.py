__author__ = 'Mustafa'

from threading import Thread
import pyttsx


class Feedback(Thread):
    def __init__(self, command=None):
        """
        Initializes the feedback object as a thread.

        :param command: The command to speak
        """
        Thread.__init__(self)

        self.command = command
        self.engine = pyttsx.init()

    def run(self):
        """
        Method to be run in a separate thread. Runs speak() to give audio feedback.

        """
        self.speak()

    def speak(self):
        """
        Method to speak a given command.

        """
        if self.command is not None:
            rate = self.engine.getProperty('rate')
            self.engine.setProperty('rate', rate - 50)  # changing speech rate
            self.engine.say(self.command)
            self.engine.runAndWait()

__author__ = "Mustafa S"


class ObjectLocation(object):
    def __init__(self, x1, y1, x2, y2):
        """
        Initializes the object with the provided two corners.
        :param x1: bottom left point x axis value
        :param y1: bottom left point y axis value
        :param x2: top right x axis value
        :param y2: top right y axis value
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.name = None  # name of the object (i.e TV, Phone, etc)
        self.description = None  # Description of the object

    def get_first_point(self):
        """
        Returns the first point coordinates (x1, y1).

        :return: The (x=int, y=int) of the first point.
        """
        return self.x1, self.y1

    def get_second_point(self):
        """
        Returns the second point coordinates (x2, y2)

        :return: The (x=int, y=int) of the second point.
        """
        return self.x2, self.y2

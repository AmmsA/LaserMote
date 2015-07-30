from unittest import TestCase
from LaserMote import LaserMote
from ObjectLocation import ObjectLocation
from Point import Point

__author__ = 'Mustafa S'


class TestLaserMote(TestCase):

    def test_is_dot_within_rois(self):
        l = LaserMote()
        o = ObjectLocation(20, 20, 40, 40)
        l.object_locations.append(o)

        self.assertEqual(o, l.is_dot_within_rois(30, 30), msg="is_dot_within_rois failed test.")
        self.assertEqual(o, l.is_dot_within_rois(20, 20), msg="is_dot_within_rois failed test.")
        self.assertEqual(o, l.is_dot_within_rois(40, 40), msg="is_dot_within_rois failed test.")
        self.assertEqual(None, l.is_dot_within_rois(50, 50), msg="is_dot_within_rois failed test.")

    def test_get_distance(self):
        l = LaserMote()
        p = Point()
        p.last_seen_x = 20
        p.last_seen_y = 20
        l.point = p

        self.assertAlmostEqual(28.2842712, l.get_distance(40, 40), places=7, msg="get_distance failed test.",
                               delta=None)

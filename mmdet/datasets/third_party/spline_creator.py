"""
Scripts to create the curves between lanes. (Splines here)
"""
import pdb

import cv2
import numpy

from mmdet.datasets.third_party import helper_scripts
from mmdet.datasets.third_party import label_file_scripts
from mmdet.datasets.third_party import dataset_constants as dc


def _draw_points(image, points, color=(255, 0, 0)):
    for point in map(tuple, points):
        cv2.circle(image, point, 2, color, 1)


def _extend_lane(lane, projection_matrix):
    """Extends marker closest to the camera

    Adds an extra marker that reaches the end of the image

    Parameters
    ----------
    lane : iterable of markers
    projection_matrix : 3x3 projection matrix
    """
    # Unfortunately, we did not store markers beyond the image plane. That hurts us now
    # z is the orthongal distance to the car. It's good enough

    # The markers are automatically detected, mapped, and labeled. There exist faulty ones,
    # e.g., horizontal markers which need to be filtered
    filtered_markers = filter(lambda x: (x['pixel_start']['y'] != x['pixel_end']['y'] and
                              x['pixel_start']['x'] != x['pixel_end']['x']),
                              lane['markers'])
    # might be the first marker in the list but not guaranteed
    closest_marker = min(filtered_markers, key=lambda x: x['world_start']['z'])

    if closest_marker['world_start']['z'] < 0:  # This one likely equals "if False"
        return lane

    # World marker extension approximation
    x_gradient = (closest_marker['world_end']['x'] - closest_marker['world_start']['x']) /\
        (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])
    y_gradient = (closest_marker['world_end']['y'] - closest_marker['world_start']['y']) /\
        (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])

    zero_x = closest_marker['world_start']['x'] - (closest_marker['world_start']['z'] - 1) * x_gradient
    zero_y = closest_marker['world_start']['y'] - (closest_marker['world_start']['z'] - 1) * y_gradient

    # Pixel marker extension approximation
    pixel_x_gradient = (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x']) /\
        (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y'])
    pixel_y_gradient = (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y']) /\
        (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x'])

    pixel_zero_x = closest_marker['pixel_start']['x'] + (716 - closest_marker['pixel_start']['y']) * pixel_x_gradient
    if pixel_zero_x < 0:
        left_y = closest_marker['pixel_start']['y'] - closest_marker['pixel_start']['x'] * pixel_y_gradient
        new_pixel_point = (0, left_y)
    elif pixel_zero_x > 1276:
        right_y = closest_marker['pixel_start']['y'] + (1276 - closest_marker['pixel_start']['x']) * pixel_y_gradient
        new_pixel_point = (1276, right_y)
    else:
        new_pixel_point = (pixel_zero_x, 716)

    new_marker = {
        'lane_marker_id': 'FAKE',
        'world_end': {'x': closest_marker['world_start']['x'],
                      'y': closest_marker['world_start']['y'],
                      'z': closest_marker['world_start']['z']},
        'world_start': {'x': zero_x, 'y': zero_y, 'z': 1},
        'pixel_end': {'x': closest_marker['pixel_start']['x'],
                      'y': closest_marker['pixel_start']['y']},
        'pixel_start': {'x': helper_scripts.ir(new_pixel_point[0]),
                        'y': helper_scripts.ir(new_pixel_point[1])}
    }
    lane['markers'].insert(0, new_marker)

    return lane


class SplineCreator():
    """
    For each lane divder
      - all lines are projected
      - linearly interpolated to limit oscillations
      - interpolated by a spline
      - subsampled to receive individual pixel values

    The spline creation can be optimized!
      - Better spline parameters
      - Extend lowest marker to reach bottom of image would also help
      - Extending last marker may in some cases be interesting too
    Any help is welcome.

    Call create_all_points and get the points in self.sampled_points
    It has an x coordinate for each value for each lane

    """
    def __init__(self, json_path):
        self.json_path = json_path
        self.json_content = label_file_scripts.read_json(json_path)
        self.lanes = self.json_content['lanes']
        self.lane_marker_points = {}
        self.sampled_points = {}  # <--- the interesting part
        self.debug_image = numpy.zeros((717, 1276, 3), dtype=numpy.uint8)

    def _sample_points(self, lane, ypp=5, between_markers=True):
        """ Markers are given by start and endpoint. This one adds extra points
        which need to be considered for the interpolation. Otherwise the spline
        could arbitrarily oscillate between start and end of the individual markers

        Parameters
        ----------
        lane: polyline, in theory but there are artifacts which lead to inconsistencies
              in ordering. There may be parallel lines. The lines may be dashed. It's messy.
        ypp: y-pixels per point, e.g. 10 leads to a point every ten pixels
        between_markers : bool, interpolates inbetween dashes

        Notes
        -----
        Especially, adding points in the lower parts of the image (high y-values) because
        the start and end points are too sparse.
        Removing upper lane markers that have starting and end points mapped into the same pixel.
        """

        # Collect all x values from all markers along a given line. There may be multiple
        # intersecting markers, i.e., multiple entries for some y values
        x_values = [[] for i in range(717)]
        for marker in lane['markers']:
            try:
                x_values[marker['pixel_start']['y']].append(marker['pixel_start']['x'])
            except IndexError:
                pdb.set_trace()

            height = marker['pixel_start']['y'] - marker['pixel_end']['y']
            if height > 2:
                slope = (marker['pixel_end']['x'] - marker['pixel_start']['x']) / height
                step_size = (marker['pixel_start']['y'] - marker['pixel_end']['y']) / float(height)
                for i in range(height + 1):
                    x = marker['pixel_start']['x'] + slope * step_size * i
                    y = marker['pixel_start']['y'] - step_size * i
                    x_values[helper_scripts.ir(y)].append(helper_scripts.ir(x))

        # Calculate average x values for each y value
        for y, xs in enumerate(x_values):
            if not xs:
                x_values[y] = -1
            else:
                x_values[y] = sum(xs) / float(len(xs))

        # In the following, we will only interpolate between markers if needed
        if not between_markers:
            return x_values  # TODO ypp

        # # interpolate between markers
        current_y = 0
        while x_values[current_y] == -1:  # skip missing first entries
            current_y += 1

        # Also possible using numpy.interp when accounting for beginning and end
        next_set_y = 0
        try:
            while current_y < 717:
                if x_values[current_y] != -1:  # set. Nothing to be done
                    current_y += 1
                    continue

                # Finds target x value for interpolation
                while next_set_y <= current_y or x_values[next_set_y] == -1:
                    next_set_y += 1
                    if next_set_y >= 717:
                        raise StopIteration

                x_values[current_y] = x_values[current_y - 1] + (x_values[next_set_y] - x_values[current_y - 1]) /\
                    (next_set_y - current_y + 1)
                current_y += 1

        except StopIteration:
            pass  # Done with lane

        return x_values

    def _lane_points_fit(self, lane):
        # TODO name and docstring
        """ Fits spline in image space for the markers of a single lane (side)

        Parameters
        ----------
        lane: dict as specified in label

        Returns
        -------
        Pixel level values for curve along the y-axis

        Notes
        -----
        This one can be drastically improved. Probably fairly easy as well.
        """
        # NOTE all variable names represent image coordinates, interpolation coordinates are swapped!
        lane = _extend_lane(lane, self.json_content['projection_matrix'])
        sampled_points = self._sample_points(lane, ypp=1)
        self.sampled_points[lane['lane_id']] = sampled_points

        return sampled_points

    def create_all_points(self,):
        """ Creates splines for given label """
        for lane in self.lanes:
            self._lane_points_fit(lane)

    def _show_lanes(self, return_only=False):
        """ For debugging spline creation only """

        gray_image = label_file_scripts.read_image(self.json_path, 'gray')
        self.debug_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        self.create_all_points()

        for _, sampled_points in self.sampled_points.items():
            _draw_points(self.debug_image, sampled_points, dc.DCOLORS[1])

        for lane_name, marker_points in self.lane_marker_points.items():
            _draw_points(self.debug_image, marker_points, dc.DICT_COLORS[lane_name])

        if not return_only:
            cv2.imshow('debug image', cv2.resize(self.debug_image, (2200, 1400)))
            cv2.waitKey(10000)

        return self.debug_image


def get_horizontal_values_for_four_lanes(json_path):
    """ Gets an x value for every y coordinate for l1, l0, r0, r1

    This allows to easily train a direct curve approximation. For each value along
    the y-axis, the respective x-values can be compared, e.g. squared distance.
    Missing values are filled with -1. Missing values are values missing from the spline.
    There is no extrapolation to the image start/end (yet).
    But values are interpolated between markers. Space between dashed markers is not missing.

    Parameters
    ----------
    json_path: str
               path to label-file

    Returns
    -------
    List of [l1, l0, r0, r1], each of which represents a list of ints the length of
    the number of vertical pixels of the image

    Notes
    -----
    The points are currently based on the splines. The splines are interpolated based on the
    segmentation values. The spline interpolation has lots of room for improvement, e.g.
    the lines could be interpolated in 3D, a better approach to spline interpolation could
    be used, there is barely any error checking, sometimes the splines oscillate too much.
    This was used for a quick poly-line regression training only.
    """

    sc = SplineCreator(json_path)
    sc.create_all_points()

    l1 = sc.sampled_points.get('l1', [-1] * 717)
    l0 = sc.sampled_points.get('l0', [-1] * 717)
    r0 = sc.sampled_points.get('r0', [-1] * 717)
    r1 = sc.sampled_points.get('r1', [-1] * 717)

    lanes = [l1, l0, r0, r1]
    return lanes

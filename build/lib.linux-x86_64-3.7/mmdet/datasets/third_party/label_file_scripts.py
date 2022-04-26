#!/usr/bin/env python3
"""
Reads and preprocesses label files, i.e. returns clean dicts
"""

import json
import os

import cv2
import numpy

from mmdet.datasets.third_party import dataset_constants as dc


def project_point(point, projection_matrix):
    """Projects 3D point into image coordinates

    Parameters
    ----------
    p1: iterable
        (x, y, z), line start in 3D
    p2: iterable
        (x, y, z), line end in 3D
    width: float
           width of marker in cm
    projection matrix: numpy.array, shape=(3, 3)
                       projection 3D location into image space

    Returns (x, y)
    """
    point = numpy.asarray(point)
    projection_matrix = numpy.asarray(projection_matrix)

    point_projected = projection_matrix.dot(point)
    point_projected /= point_projected[2]

    return point_projected


def project_lane_marker(p1, p2, width, projection_matrix, color, img):
    """ Draws a marker by two 3D points (p1, p2) in 2D image space

    p1 and p2 are projected into the image space using a given projection_matrix.
    The line is given a fixed width (in cm) to be drawn. Since the marker width
    is given for the 3D space, the closer edge will be thicker in the image.
    The color can be freely set, e.g. according to lane association.

    Parameters
    ----------
    p1: iterable
        (x, y, z), line start in 3D
    p2: iterable
        (x, y, z), line end in 3D
    width: float
           width of marker in m, default=0.1 m
    projection matrix: numpy.array, shape=(3, 3)
                       projection 3D location into image space
    color: int or tuple
           color of marker, e.g. 255 or (0, 255, 255) in (b, g, r)
    img: numpy.array (dtype=numpy.uint8)
         Image array to draw the marker

    Notes
    ------
    You can't draw colored lines into a grayscale image.
    """
    p1 = numpy.asarray(p1)
    p2 = numpy.asarray(p2)

    p1_projected = project_point(p1, projection_matrix)
    p2_projected = project_point(p2, projection_matrix)

    points = numpy.zeros((4, 2), dtype=numpy.float32)
    shift = 0
    # shift_multiplier = static_cast<double>(1 << shift)
    shift_multiplier = 1  # simplified

    projection_matrix = numpy.asarray(projection_matrix)
    projected_half_width1 = projection_matrix[0, 0] * width / p1[2] / 2.0
    points[0, 0] = (p1_projected[0] - projected_half_width1) * shift_multiplier
    points[0, 1] = p1_projected[1] * shift_multiplier
    points[1, 0] = (p1_projected[0] + projected_half_width1) * shift_multiplier
    points[1, 1] = p1_projected[1] * shift_multiplier

    projected_half_width2 = projection_matrix[0, 0] * width / p2[2] / 2.0
    points[2, 0] = (p2_projected[0] + projected_half_width2) * shift_multiplier
    points[2, 1] = p2_projected[1] * shift_multiplier
    points[3, 0] = (p2_projected[0] - projected_half_width2) * shift_multiplier
    points[3, 1] = p2_projected[1] * shift_multiplier

    points = numpy.round(points).astype(numpy.int32)

    if not points[0, 1] == points[3, 1]:
        try:  # difference in cv2 versions
            aliasing = cv2.LINE_AA
        except AttributeError:
            aliasing = cv2.CV_AA
        cv2.fillConvexPoly(img, points, color, aliasing, shift)
        cv2.fillConvexPoly(img, points, color, aliasing, shift)


def __get_base_name(input_path):
    """ /foo/bar/test/folder/image_label.ext --> test/folder/image_label.ext """
    return '/'.join(input_path.split('/')[-3:])


def read_image(json_path, image_type='gray'):
    """ Reads image corresponding to json file

    Parameters
    ----------
    json_path: str
               path to json file / label
    image_type: str
                type of image to read, either 'gray' or 'color'

    Returns
    -------
    numpy.array
        Image corresponding to image file

    Raises
    ------
    ValueError
        If image_type is neither 'gray' nor 'color'
    IOError
        If image_path does not exist. The image folder may not exist
        or may not be set in dataset_constants.py
    """
    # NOTE The function is built like this because extensions offer access to other types
    base_name = __get_base_name(json_path)
    if image_type == 'gray':
        image_path = os.path.join(dc.GRAYSCALE_IMAGES, base_name.replace('.json', '_gray_rect.png'))
        imread_code = cv2.IMREAD_GRAYSCALE
    elif image_type == 'color':
        image_path = os.path.join(dc.COLOR_IMAGES, base_name.replace('.json', '_color_rect.png'))
        imread_code = cv2.IMREAD_COLOR
    else:
        ValueError('Unknown image_type: {}'.format(image_type))

    if not os.path.exists(image_path):
        raise IOError(
            'Image does not exist: {}\n. Did you set dataset_constants.py?'.format(image_path))
    return cv2.imread(image_path, imread_code)


def _fix_json(json_string):
    """ The 'json' output of the label creation tool does not natively
    work with Python's json. This one is a quick fix to correct these issues.
    """
    # NOTE should be applied to label files directly
    json_string.replace('",\n\t\t\t\t"lane_marker": {', '",\n\t\t\t\t"markers": [')
    json_lines = json_string.split('\n')
    json_lines.pop(1)
    json_lines.pop(-1)
    json_lines.pop(-1)
    for i in range(len(json_lines)):
        if json_lines[i] == '\t\t"lanes": {':
            json_lines[i] = '\t\t"lanes": ['
        elif json_lines[i] == '\t\t\t"lane": {':
            json_lines[i] = '\t\t\t{'
        elif json_lines[i] == '\t\t}':
            json_lines[i] = '\t\t]'

        # now inner lane markers
        if json_lines[i] == '\t\t\t\t"lane_marker": {':
            json_lines[i] = '\t\t\t\t{'
    json_string = '\n'.join(json_lines)

    # global stop of lists
    json_string = json_string.replace('",\n\t\t\t\t{', '",\n\t\t\t\t"markers": [\n\t\t\t\t{')
    json_string = json_string.replace('\t\t\t\t}\n\t\t\t}', '\t\t\t\t}]\n\t\t\t}')
    return json_string


def _filter_lanes_by_size(label, min_height=40):
    """ May need some tuning """
    filtered_lanes = []
    for lane in label['lanes']:
        lane_start = min([int(marker['pixel_start']['y']) for marker in lane['markers']])
        lane_end = max([int(marker['pixel_start']['y']) for marker in lane['markers']])
        if (lane_end - lane_start) < min_height:
            continue
        filtered_lanes.append(lane)
    label['lanes'] = filtered_lanes


def _filter_few_markers(label, min_markers=2):
    """Filter lines that consist of only few markers"""
    filtered_lanes = []
    for lane in label['lanes']:
        if len(lane['markers']) >= min_markers:
            filtered_lanes.append(lane)
    label['lanes'] = filtered_lanes


def _fix_lane_names(label):
    """ Given keys ['l3', 'l2', 'l0', 'r0', 'r2'] returns ['l2', 'l1', 'l0', 'r0', 'r1']"""

    # Create mapping
    l_counter = 0
    r_counter = 0
    mapping = {}
    lane_ids = [lane['lane_id'] for lane in label['lanes']]
    for key in sorted(lane_ids):
        if key[0] == 'l':
            mapping[key] = 'l' + str(l_counter)
            l_counter += 1
        if key[0] == 'r':
            mapping[key] = 'r' + str(r_counter)
            r_counter += 1
    for lane in label['lanes']:
        lane['lane_id'] = mapping[lane['lane_id']]


def read_json(json_path, min_lane_height=20):
    """ Reads and cleans label file information by path"""
    with open(json_path, 'r') as jf:
        label_content = json.load(jf)

    _filter_lanes_by_size(label_content, min_height=min_lane_height)
    _filter_few_markers(label_content, min_markers=2)
    _fix_lane_names(label_content)

    content = {
        'projection_matrix': label_content['projection_matrix'],
        'lanes': label_content['lanes']
    }

    for lane in content['lanes']:
        for marker in lane['markers']:
            for pixel_key in marker['pixel_start'].keys():
                marker['pixel_start'][pixel_key] = int(marker['pixel_start'][pixel_key])
            for pixel_key in marker['pixel_end'].keys():
                marker['pixel_end'][pixel_key] = int(marker['pixel_end'][pixel_key])
            for pixel_key in marker['world_start'].keys():
                marker['world_start'][pixel_key] = float(marker['world_start'][pixel_key])
            for pixel_key in marker['world_end'].keys():
                marker['world_end'][pixel_key] = float(marker['world_end'][pixel_key])
    return content

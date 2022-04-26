"""
Contains paths to dataset, working directory, and tfrecords
"""
import os
import time


# TODO Set path
COLOR_IMAGES = "..../unsupervised_llamas/color_images"
GRAYSCALE_IMAGES = "..../unsupervised_llamas/grayscale_images"
LABELS = "..../unsupervised_llamas/labels"

# TODO set path
WORKING_DIRECTORY = ".../some_path/markers/"
TFRECORDS_FOLDER = os.path.join(WORKING_DIRECTORY, 'processed_data/tfrecords')
TRAINED_NETS = os.path.join(WORKING_DIRECTORY, 'trained_nets')

# A specific training directory based on time
TRAIN_DIRECTORY = os.path.join(TRAINED_NETS, time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))

NUM_TRAIN_IMAGES = 58269
NUM_VALID_IMAGES = 20844
NUM_TEST_IMAGES = 20929

# Multi-class segmentation colors for the individual lanes
# The names are based on the camera location, e.g. the markers
# from r2 divide the first lane to the right from the second to the right
DCOLORS = [(110, 30, 30), (75, 25, 230), (75, 180, 60), (200, 130, 0), (48, 130, 245), (180, 30, 145),
           (0, 0, 255), (24, 140, 34), (255, 0, 0), (0, 255, 255),  # the main ones
           (40, 110, 170), (200, 250, 255), (255, 190, 230), (0, 0, 128), (195, 255, 170),
           (0, 128, 128), (195, 255, 170), (75, 25, 230)]
LANE_NAMES = ['l7', 'l6', 'l5', 'l4', 'l3', 'l2',
              'l1', 'l0', 'r0', 'r1',
              'r2', 'r3', 'r4', 'r5',
              'r6', 'r7', 'r8']
DICT_COLORS = dict(zip(LANE_NAMES, DCOLORS))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from skimage.feature import hog
from lesson_functions import *
import pickle
import scipy.misc
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from IPython.display import HTML

class Scene():
    def __init__(self):
        self.frame_number = 0
        self.car_count = 0

        # Classifier
        self.classifier = []
        self.X_scaler = []

        # Current bounding boxes
        self.bbox = []

        # Keep track of the bounding boxes over frames
        self.moving_hotbox = []
        # Number of frames to include in the moving average including current frame
        self.n_moving_frames = 3

        # Reasonable Search Region
        # Initialize low/high so they will be overridden with good values
        self.min_x = 1280
        self.max_x = 0
        self.min_y = 720
        self.max_y = 0

        # Initialize to a reasonable image size
        self.width = 1280
        self.height = 720
        self.half_height = 360

        # Controls how often to perform grid search
        self.perform_search = True
        self.search_count = 0

        # Moving average of heatmaps
        self.heat0 = []   # heatmap for current frame
        self.heat1 = []   # heatmap for last frame
        self.heat2 = []   # heatmap for 2 frames ago

        # HOG parameters
        self.color_space = 'HSV'
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.spatial_size = (16, 16)
        self.hist_bins = 48
        self.spatial_feat = True
        self.hist_feat = True
        self.hog_feat = True

        self.debug = False

    # Read in the classifier and X_scaler
    def get_model(self, filename):
        pkl_file = open(filename, 'rb')
        self.clf, self.X_scaler = pickle.load(pkl_file)
        pkl_file.close()

    # Only need to set this once because the image size
    # is not going to change for the same video file
    def get_image_size(self, img):
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.half_height = np.int(self.height/2)

    def initialize_moving_heatmap(self, img):
        self.heat0 = np.ones_like(img[:,:,0]).astype(np.float)*5
        self.heat1 = np.ones_like(img[:,:,0]).astype(np.float)*5
        self.heat2 = np.ones_like(img[:,:,0]).astype(np.float)*5

    # Calculate weighted average of last 3 frames' heatmaps
    def calc_moving_heatmap(self):
        weighted_heatmap = self.heat0 + (self.heat1 * 2/3).astype(int) + (self.heat2 / 3).astype(int)

        # Age heatmaps by 1 frame
        self.heat1 = self.heat0
        self.heat2 = self.heat1

        return weighted_heatmap


    def single_img_features_append_channels(self, img):
        img_features = []

        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)

        if self.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)
            # 4) Append features to list
            img_features.append(spatial_features)

        if self.hist_feat == True:
            hist_features = color_hist(feature_image, nbins=self.hist_bins)
            # 6) Append features to list
            img_features.append(hist_features)

        if self.hog_feat == True:
            # Extract every channel and append to features
            hog_features = get_hog_features(feature_image[:, :, 0], self.orient,
                                            self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
            img_features.append(hog_features)

            hog_features = get_hog_features(feature_image[:, :, 1], self.orient,
                                            self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
            img_features.append(hog_features)

            hog_features = get_hog_features(feature_image[:, :, 2], self.orient,
                                            self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
            img_features.append(hog_features)

        return np.concatenate(img_features)

    def search_windows_extended(self, img, windows):
        on_windows = []

        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = self.single_img_features_append_channels(test_img)
            test_features = self.X_scaler.transform(np.array(features).reshape(1, -1))
            prediction = self.clf.predict(test_features)

            if prediction == 1:
                on_windows.append(window)

        return on_windows

    # Find the Heat Map from all the bounding boxes
    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        return heatmap

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0

        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Save all the bounding boxes
        total_bbox = []

        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            total_bbox += [bbox]

            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 4)

        return img, total_bbox

    # Define a function where the xy_window is larger closer to the bottom of the image
    # and smaller as it moves up the image
    def slide_window_adaptive_proportion(self, img):


        mid_x = np.int(self.width / 2)
        #end_y = np.int(self.height * 0.9)

        # Search windows get slightly bigger as we move down the image
        for size in [80, 90, 100, 120, 150]:
            #print("size: ", size)

            if size == 80:
                windows = slide_window(img, x_start_stop=[mid_x - 300, self.width],
                                       y_start_stop=[self.half_height, np.int(self.height * 0.69)],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows = windows
            elif size == 90:
                windows = slide_window(img, x_start_stop=[mid_x - 320, self.width - 20],
                                       y_start_stop=[self.half_height, np.int(self.height * 0.75)],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows
            elif size == 100:
                windows = slide_window(img, x_start_stop=[mid_x - 340, 1050],
                                       y_start_stop=[self.half_height + 20, np.int(self.height * 0.80)],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows
            elif size == 120:
                windows = slide_window(img, x_start_stop=[mid_x - 360, self.width],
                                       y_start_stop=[self.half_height + 20, np.int(self.height * 0.90)],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows
            elif size == 150:
                windows = slide_window(img, x_start_stop=[mid_x - 380, self.width],
                                       y_start_stop=[self.half_height + 50, np.int(self.height * 0.92)],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows
            elif size == 200:
                windows = slide_window(img, x_start_stop=[mid_x - 400, self.width],
                                       y_start_stop=[self.half_height + 80, np.int(self.height * 0.92)],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows

        return all_windows

    def slide_window_recent_region(self, img):
        x_offset = 25
        y_offset = 10

        # Starting y-value cannot be higher than the horizon, mid-point of height
        start_y = np.maximum(self.min_y - y_offset, self.half_height)

        # Ending y_value cannot be lower than the hood of the car
        end_y = np.minimum(self.max_y + y_offset, np.int(self.height * 0.92))

        for size in [64, 80, 90, 100, 120, 150]:
            if size == 64:
                windows = slide_window(img, x_start_stop=[self.min_x - x_offset, self.max_x + x_offset],
                                       y_start_stop=[start_y, end_y],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows = windows
            if size == 80:
                windows = slide_window(img, x_start_stop=[self.min_x - x_offset, self.max_x + x_offset],
                                       y_start_stop=[start_y, end_y],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows
            elif size == 90:
                windows = slide_window(img, x_start_stop=[self.min_x - x_offset, self.max_x + x_offset],
                                       y_start_stop=[start_y, end_y],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows
            elif size == 100:
                windows = slide_window(img, x_start_stop=[self.min_x - x_offset, self.max_x + x_offset],
                                       y_start_stop=[start_y, end_y],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows
            elif size == 120:
                windows = slide_window(img, x_start_stop=[self.min_x - x_offset, self.max_x + x_offset],
                                       y_start_stop=[start_y, end_y],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows
            elif size == 150:
                windows = slide_window(img, x_start_stop=[self.min_x - x_offset, self.max_x + x_offset],
                                       y_start_stop=[start_y, end_y],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows
            elif size == 200:
                windows = slide_window(img, x_start_stop=[self.min_x - x_offset, self.max_x + x_offset],
                                       y_start_stop=[self.min_y - y_offset, self.max_y + y_offset],
                                       xy_window=(size, size), xy_overlap=(0.8, 0.8))
                all_windows += windows

        return all_windows


    # Find a reasonable search region for quick scan for cars
    # To be used between slide_window_adaptive_proportion() full's scan of the image
    def recent_search_region(self):
        # Only need to compare to find new search region
        # if there are bounding boxes found
        if len(self.bbox) > 0:
            for idx, item in enumerate(self.bbox):
                    x1 = item[0][0]
                    y1 = item[0][1]
                    x2 = item[1][0]
                    y2 = item[1][1]

                    self.min_x = np.minimum(self.min_x, x1)
                    self.min_x = np.minimum(self.min_x, x2)
                    self.max_x = np.maximum(self.max_x, x1)
                    self.max_x = np.maximum(self.max_x, x2)

                    self.min_y = np.minimum(self.min_y, y1)
                    self.min_y = np.minimum(self.min_y, y2)
                    self.max_y = np.maximum(self.max_y, y1)
                    self.max_y = np.maximum(self.max_y, y2)


    def pipeline(self, img):
        self.frame_number += 1

        # Full Search for 1st and every nth frame
        if self.frame_number == 1:
            self.get_image_size(img)
            apt_windows = self.slide_window_adaptive_proportion(img)

            self.initialize_moving_heatmap(img)
            self.search_count = 0
            self.perform_search = True

            print("full search ... ")

        elif np.remainder(self.frame_number, 6) > 0: # Do a smaller search region

            # Variable to control when to perform a grid search
            if self.car_count > 0:
                self.search_count = 1
                self.perform_search = True
            else:
                self.search_count += 1
                if np.remainder(self.search_count, 2) == 0:
                    self.perform_search = True
                    self.search_count = 0
                else:
                    self.perform_search = False

            print("car_count: ", self.car_count, "perform_search: ", str(self.perform_search), "search_count: ", self.search_count)


            # If no bounding boxes are found yet then min_x has not changed
            # Go back to full scan
            if self.perform_search:
                if self.min_x == 1280:
                    apt_windows = self.slide_window_adaptive_proportion(img)
                    print("full search ... ")
                else:
                    # Create a recent search region between full scan of image
                    apt_windows = self.slide_window_recent_region(img)
                    print("short search: ")
            else:
                print("Skip this frame ... no grid search")

        else:
            self.search_count = 0
            self.perform_search = True
            apt_windows = self.slide_window_adaptive_proportion(img)
            print("full search ... ")

        # If last frame no car was found then search every other frame for speed consideration
        if self.perform_search:
            hot_windows = self.search_windows_extended(img, apt_windows)

            # Apply heatmap
            heat = np.zeros_like(img[:, :, 0]).astype(np.float)
            self.heat0 = self.add_heat(heat, hot_windows)

            # Consider heatmaps from last 2 frames
            weighted_heatmap = self.calc_moving_heatmap()
            threshold_heatmap = self.apply_threshold(weighted_heatmap, 15)

            labels = label(threshold_heatmap)
            self.car_count = labels[1]

        # Only draw boxes & find a smaller search region if car is found
        if self.perform_search or self.car_count > 0:
            # Bounding Boxes
            draw_img_boxes, self.bbox = self.draw_labeled_bboxes(np.copy(img), labels)
            self.recent_search_region()
        else:
            draw_img_boxes = np.copy(img)

        print("frame_number: ", self.frame_number, " car_count: ", self.car_count, " search region: (",
              self.min_x, self.min_y, "), ( ", self.max_x, self.max_y, ")")

        return draw_img_boxes

# Initialize Class and read in the pre-trained model
print("Processing Starts ... ")
scene = Scene()
scene.get_model('SVM_orient9_cellsPerBlock2_histBins48.pkl')

# Project Video
video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")

t0 = time.time()
clip_output = clip1.fl_image(scene.pipeline)
clip_output.write_videofile(video_output, audio=False)

print("Processing Time: ", round(time.time() - t0, 2))

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
from sklearn.model_selection import train_test_split


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features_append_channels(imgs, color_space='RGB', spatial_size=(32, 32),
                                     hist_bins=32, orient=9,
                                     pix_per_cell=8, cell_per_block=2,
                                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        # orig: image = mpimg.imread(file)

        image = scipy.misc.imread(file)

        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Extract every channel and append to features
            hog_features = get_hog_features(feature_image[:, :, 0], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)

            hog_features = get_hog_features(feature_image[:, :, 1], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)

            hog_features = get_hog_features(feature_image[:, :, 2], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Read data
# Read in cars and notcars
cars_list = glob.glob("train_images/vehicles/*/*.png")
not_cars_list = glob.glob("train_images/non-vehicles/*/*.png")

cars = []
notcars = []

for car in cars_list:
    cars.append(car)

for not_car in not_cars_list:
    notcars.append(not_car)

color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
spatial_size = (16, 16) # Spatial binning dimensions
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

for cell_per_block in [2, 4]:
    for orient in [9, 12]:
        for hist_bins in [32, 48]:
            y_start_stop = [None, None] # Min and max in y to search in slide_window()

            car_features = extract_features_append_channels(cars, color_space=color_space, \
                                    spatial_size=spatial_size, hist_bins=hist_bins, \
                                    orient=orient, pix_per_cell=pix_per_cell, \
                                    cell_per_block=cell_per_block, \
                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

            notcar_features = extract_features_append_channels(notcars, color_space=color_space, \
                                    spatial_size=spatial_size, hist_bins=hist_bins, \
                                    orient=orient, pix_per_cell=pix_per_cell, \
                                    cell_per_block=cell_per_block, \
                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

            X = np.vstack((car_features, notcar_features)).astype(np.float64)

            # Fit a per-column scaler
            X_scaler = StandardScaler().fit(X)

            # Apply the scaler to X
            scaled_X = X_scaler.transform(X)

            # Define the labels vector
            y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

            # Split up data into randomized training and test sets
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=0.2, random_state=rand_state)

            print('Using: ', orient, 'orientations', pix_per_cell,
                  'pixels per cell and ', cell_per_block, 'cells per block ',
                   hist_bins, 'hist_bins => feature vector length: ', len(X_train[0]))

            # Use a linear SVC
            svc = LinearSVC()
            svc_model = CalibratedClassifierCV(svc)

            # Check the training time for the SVC
            t=time.time()
            svc_model.fit(X_train, y_train)

            t2 = time.time()
            print('   Seconds to train SVC: ' , round(t2-t, 2))

            # Check the score of the SVC
            print('   Train Accuracy of SVC: ', round(svc_model.score(X_train, y_train), 4))
            print('   Test  Accuracy of SVC:  ', round(svc_model.score(X_test, y_test), 4))

            # Check the prediction time for a single sample
            i = rand_seed = np.random.randint(0, 1000)
            t0 = time.time()
            prediction = svc_model.predict(X_test[i].reshape(1, -1))
            prob = svc_model.predict_proba(X_test[i].reshape(1, -1))

            print("      Prediction time with SVC: ", time.time()-t0)
            print("      Label ", int(y_test[i]), "Prediction {}".format(prediction))
            print("      Prob {}".format(prob))

            # Save Model
            filename = 'SVM_orient' + str(orient) + '_cellsPerBlock' + str(cell_per_block) + '_histBins' + str(hist_bins) + '.pkl'
            with open(filename, 'wb') as fp:
                pickle.dump([svc_model, X_scaler], fp)

            fp.close()

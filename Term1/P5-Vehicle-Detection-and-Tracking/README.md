##Writeup Template

__Author: Annie Flippo__

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/car_not_car.png
[image2]: ./writeup_images/HOG_example.png
[image3]: ./writeup_images/sliding_windows.png
[image4]: ./writeup_images/sliding_window_detection.png
[image5]: ./writeup_images/sample_pipeline_test_images.png
[image6]: ./writeup_images/test_image_label.png
[image7]: ./writeup_images/test_image_bbox.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the __train_model.py__ from line 74 to 75.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces.  I visually did not see a big  difference between the color spaces.  I thought I let the classifier inform me which color space is better by its predictive power.  I used reasonable values for orientations (9 and 12), cells_per_block (2 and 4) and hist_bins (32, 48) to run all possible combinations of these parameters through `get_hog_features( )`` function which called `skimage.hog()`

 I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]


The difference in HOG features between car and not-car look very promising.

####2. Explain how you settled on your final choice of HOG parameters.

As mentioned in Section 1, I used reasonable values for orientations (9 and 12), cells_per_block (2 and 4) and hist_bins (32, 48) to run all possible combinations of these parameters to run through my classifier to see which of the combination gave me the highest test accuracy while using as few features as possible.   In train_model.py, starting line 92 I run loops to vary `cell_per_block`, `orient` and `hist_bins` to extract feature and train my classifier.   

See here the output of these different combinations of parameters and its test accuracy. 

```
$python train_model.py
Using:  9 orientations 8 pixels per cell and  2 cells per block  32 hist_bins => feature vector length:  6156
   Seconds to train SVC:  19.71
   Train Accuracy of SVC:  1.0
   Test  Accuracy of SVC:   0.993
      Prediction time with SVC:  0.0009789466857910156
      Label  1 Prediction [ 1.]
      Prob [[  3.81635746e-04   9.99618364e-01]]
Using:  9 orientations 8 pixels per cell and  2 cells per block  48 hist_bins => feature vector length:  6204
   Seconds to train SVC:  6.5
   Train Accuracy of SVC:  1.0
   Test  Accuracy of SVC:   0.9932
      Prediction time with SVC:  0.0008289813995361328
      Label  0 Prediction [ 0.]
      Prob [[  9.99943088e-01   5.69122070e-05]]
Using:  12 orientations 8 pixels per cell and  2 cells per block  32 hist_bins => feature vector length:  7920
   Seconds to train SVC:  7.81
   Train Accuracy of SVC:  1.0
   Test  Accuracy of SVC:   0.9918
      Prediction time with SVC:  0.0007729530334472656
      Label  0 Prediction [ 0.]
      Prob [[ 0.98958434  0.01041566]]
Using:  12 orientations 8 pixels per cell and  2 cells per block  48 hist_bins => feature vector length:  7968
   Seconds to train SVC:  7.8
   Train Accuracy of SVC:  1.0
   Test  Accuracy of SVC:   0.9958
      Prediction time with SVC:  0.0007638931274414062
      Label  0 Prediction [ 0.]
      Prob [[  9.99648172e-01   3.51828474e-04]]
Using:  9 orientations 8 pixels per cell and  4 cells per block  32 hist_bins => feature vector length:  11664
   Seconds to train SVC:  24.48
   Train Accuracy of SVC:  1.0
   Test  Accuracy of SVC:   0.9885
      Prediction time with SVC:  0.0008249282836914062
      Label  0 Prediction [ 0.]
      Prob [[  9.99150335e-01   8.49664960e-04]]
Using:  9 orientations 8 pixels per cell and  4 cells per block  48 hist_bins => feature vector length:  11712
   Seconds to train SVC:  28.22
   Train Accuracy of SVC:  1.0
   Test  Accuracy of SVC:   0.9896
      Prediction time with SVC:  0.0014770030975341797
      Label  1 Prediction [ 1.]
      Prob [[  6.46387331e-06   9.99993536e-01]]
Using:  12 orientations 8 pixels per cell and  4 cells per block  32 hist_bins => feature vector length:  15264
   Seconds to train SVC:  83.31
   Train Accuracy of SVC:  1.0
   Test  Accuracy of SVC:   0.9927
      Prediction time with SVC:  0.0008671283721923828
      Label  1 Prediction [ 1.]
      Prob [[  1.05959030e-04   9.99894041e-01]]
Using:  12 orientations 8 pixels per cell and  4 cells per block  48 hist_bins => feature vector length:  15312
   Seconds to train SVC:  67.46
   Train Accuracy of SVC:  1.0
   Test  Accuracy of SVC:   0.9921
      Prediction time with SVC:  0.0021190643310546875
      Label  1 Prediction [ 1.]
      Prob [[  2.21560197e-06   9.99997784e-01]]
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the end, I chose a linear SVM using 9 orientations, 8 pixels per cell, 2 cells per block and 48 hist_bins which yielded 6204 features vector length and a test accuracy of approximately 99.32%.  I finally settled on color space of `HSV`.  I tried `RGB` which seemed a little less accurate and I didn't see a big difference using `YUV`.   In the earlier step when I ran the linear SVM classifier with each parameter combination, I also saved my model out into a pickle file in case I found a winner and won't have to train it again.

In `train_model.py` line 99 and 105 where I call `extract_features_append_channels` which in turned calls `get_hog_features`.
The color features are select from starting line 32 in `train_model.py`

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to only look at the image below the horizon starting at mid-image and downward but above the hood of the car.  I don't care about flying cars yet!  This was an iterative process for me.  I first tested small windows closer to the horizon and I slowly increase the window size as I move down the image.  I also tried very small windows which gave me lots of false positives during processing of the video.  Here are some small windows closer to the horizon and larger windows lower on the image.  You might not notice the window size due to 90% overlap of each window.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I used HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example from the test images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my YouTube video result](https://youtu.be/tWO4I8ITwqw)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I scan the bottom half of the image using my adaptive sliding windows in find_cars.py on line 196 in function `slide_window_adaptive_proportion( )`.  Notice that some images gave false positive in identifying part of the road as car.  I created heatmap with appropriate thresholding to remove false positive identification on line 364-365 calling `add_heat( )`.  Additionally, I used a weighted average of heatmap of current frame, last frame and 2 frames ago to further remove false positives in a custom function `calc_moving_heatmap( )` from line 88-95 in find_cars.py.  

Finally, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing car feature predictions, the heatmap from a series of frames of video, the result of `label()` and the bounding boxes then overlaid on the test_images provided:

### Here are six test images and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap for test_image/test6.jpg:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the test_image/test6.jpb:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline would likely to fail around traffic signs.  I had a lot of false positives around the green signs (miles to the next city) signs.  I had to do a lot of thresholding tests to remove that.  What I noticed is that the training images are mostly from European cars and road signs.  I would add more North American road signs to make the classifier more robust.

I've found that using very small search windows will yield in a lot of false positives.  I spent a lot of time fine-tuning the sliding windows from fixed sizes to a more variable sizes.  I've dissected the test images into different regions some near the horizon and some closer to the bottom of the image for close inspections.  What I've found is that you want the car features to be mostly wholely contained within the search window as much as possible.  That's why I used 64 x 64 pixels search window near mid-point and gradually increased to 200 x 200 pixels near the bottom.

Next, I tried to speed up the processing of the video.  If I did a full scan of the bottom of the image for every frame, it will take almost 4 hours to process the entire project_video.mp4.  I decided to experiment with different techniques to reduce processing time.  In the end, I performed the following:

- If there were no car detected in a full-scan sliding window in a frame, I would skip a scan in the next window.  The idea is that at 25 frames per second, I wouldn't see any noticeable change in car detection from frame to frame but I would reduce processing time by half.

- If there were cars detected in a full-scan sliding window in a specific frame, I would do a more reasonable search in a smaller region in the next frame.  Mainly, I did this to track the movement of the car but I wouldn't have to perform a full-scan which again reduce processing time.  I wrote `slide_window_recent_region( )` in find_cars.py starting line 239.

Lastly, I think performing image search of features using just computer vision is pretty slow.  It's probably too slow for real-time processing of object detection in a real driving car application.  Ultimately I would try using some deep learning techniques that was discussed in our Slack channel and other published techniques such as YOLO (You Only Look Once) or SSD (Single Shot Detection).




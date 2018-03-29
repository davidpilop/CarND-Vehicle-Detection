# Project 05 - Vehicle Detection and Tracking
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this vehicle detection and tracking project, we detect in a video pipeline, potential boxes, via a sliding window, that may contain a vehicle by using a Support Vector Machine Classifier for prediction to create a heat map. The heat map history is then used to filter out false positives before identification of  vehicles by drawing a bounding box around it.

## The goals / steps of this project are the following:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
    * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
    * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Files
* `P5.ipynb` : main code for this project
* `utils.py`: contains the main functions of the project
* `output_images` : images used to illustrate the steps taken to complete this project
* `test_images` : images used to test the functions in this project
* `README.md` : readme file

---

### Loading and Visualizing the data
For this project I used the `vehicle` (labeled as `cars`) and `non-vehicle` (labeled as `notcars`) datasets provided by [Udacity](https://github.com/udacity/CarND-Vehicle-Detection). Below is are 8 random images from the `vehicle` and `non-vehicle` datasets

![data](,/output_images/data.png "data")

### Defining a function to return HOG features and visualization
The code for extracting HOG features from an image is defined by the method `get_hog_features` which uses the `hog()` function from the [scikit-image](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog) package. Below is the visualization of the function.

![HOG](,/output_images/HOG.png "HOG")

HOG parameters are mostly tuned during sliding window search. I used these parameters
- orient: 32
- pixels_per_cell: 16, testing shows 16 better than 8
- cells_per_block: 2

### Defining a function to compute Color Histogram features and visualizing the results
The `color_hist` function computes Color Histogram features labeled `hist_features`. This function returns concatenated color channels by default and separate color channels if the `vis == True` flag is called. Below is the visualization of the 'R' 'G' and 'B' channels from a random `car_image`.

![hist](/output_images/hist.png "hist")

The `bin_spatial` function is useful for extracting color features from low resolution images. Below is an example of spatially binned color features extracted from an image before and after resizing.

![bin_spatial](/output_images/spatila_bin.png "bin_spatial")

### Defining a function to extract features from a list of images
The `extract_features` function accepts a list of image paths and produces a flattened array of features returned by the `bin_spatial`, `color_hist` and `get_hog_features` functions. This functions are called by `single_img_features` function. Each functions can be extracted individually or all at once thanks to flags.

### Sliding Window Implementation
The `slide_window` function generates a list of windows which will then be passed to draw boxes. Previously, for each window, I extracted features and had the model to make the prediction. Object in different positions will have different size because the further the distance, the smaller the object. The main drawback was the slow speed because it has to calculate the HOG feature for each window. Below is an illustration of the `slide_window` function with adjusted `y_start_stop` values [400, 656].
This function

![sliding_windows](/output_images/sliding_windows.png "sliding_windows")

The extracted features are passed on to the `search_windows` function which searches windows for matches defined by the classifier.

### Adding Heatmaps and Bounding Boxes
The `add_heat` function iterates through list of bboxes and add += 1 for all pixels inside each bboxm assuming each "box" takes the form ((x1, y1), (x2, y2)).
The `apply_threshold` function defines how many search boxes have to overlap for the pixels to be counted as "hot" with the purpose of eliminating the errors.
The `draw_labeled_bboxes` function takes in the "hot" pixel values from the image and converts them into labels then draws bounding boxes around those labels.

![bounding](/output_images/bounding.png "bounding")

### Defining a function that can extract features using HOG sub-sampling and make predictions
The `find_cars` function extracts the HOG and color features, scales them and then makes predictions. Using multiple scale values allows for more accurate predictions. I have combined scales of `1.0, 1.5` and `2.0` with their own `ystart` and `ystop` values to lower the ammount of false-postive search boxes.

![found](/output_images/found.png "found")

### Conclusion
I would like to try using deep-learning for vehicle recognition beacause i'm not happy with the result. There are a lot of false positives and maybe using deep-learning we could reduce this. However the result will be slower than using SVM, so it is possible that this is the best method under certain circumstances.

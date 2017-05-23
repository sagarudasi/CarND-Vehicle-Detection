
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/11.png
[image2]: ./output_images/extra11.png
[image3]: ./output_images/hog.PNG
[image4]: ./output_images/hls.PNG
[image5]: ./output_images/small_win.PNG
[image6]: ./output_images/medium_win.PNG
[image7]: ./output_images/large_win.PNG
[image8]: ./output_images/test_image.PNG
[image9]: ./output_images/test_hls.PNG
[image10]: ./output_images/combined.png
[image11]: ./output_images/heatmap.png
[image12]: ./output_images/final_bboxes.PNG

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

##### Analysis of the dataset

The dataset consits of images of car and non-car (extra) objects. Futhermore, the dataset available is from KITTI and GTI databases. Following is the example of car and non-car image respectively -

![Car image][image1] 
![Non car image][image2]

All the images are 64x64 pixel color images stored in PNG image format.

There are about - 8700 + Car images and 8900 + Non car images.

So the dataset is sufficiently distributed i.e. we have almost equal number of images of each category.

##### Extracting HOG features 

For extracting the Histogram of Oriented Gradients, skimage library provides a nice helper function called hog with the following definition -
```python
  hog(img, orientations, pixels_per_cell, cells_per_block, visualise, feature_vector) 
```

I am using the above function to generate hog features for all the input channels of the image. 
The configuration I am using is as following - 
orient=9
pix_per_cell=8
cell_per_block=2
hog_channel='ALL'

The following image shows the visualization of hog transform performed on the dataset vehicle image -

![Hog transform][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I used spatial and color histogram features apart from the hog features.
The very first thing was to decide over colorspace. Options available were RGB, HLS, HSV, YCrCb etc.
As learnt from the previous projects, RGB is not good in light and dark patches and the color separation is not efficient.
HLS color space was working fine in the Advanced lane detection project so I decided to use the same.

Following is the visualization when BGR is converted to HLS -
![HLS color space][image4]

Also the same color space was used to extract other features.

Next thing I tried was using all the 0,1 and 2 channels for the HOG. Obvisiously, having more features will help fit the machine better but comes with a speed trade-off.
Since we are running the pipeline offline, I decided to stick with using the all the channels for HOG. 
The other parameters for HOG and spatial bin were kept default.

For color histogram, since I decided to stick to HLS and I was reading the images using cv2.imread, the range required was 0 - 255.

Following is the final configuration used - 

color_space: 'HLS'
spatial_size: (32, 32)
hist_bins: 32
hist_range: (0,255)
orient: 9
pix_per_cell: 8
cell_per_block: 2
hog_channel: 'ALL'

File common.py contains the following functions which are used to perform above operations - 
convert_colorspace
get_hog_features
bin_spatial
color_hist

Also two different functions were written to extract features from single image and multiple images respectively - 
single_img_features
multiple_img_features


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the classifier is in svc_main file from line number 41 to 80
First I combined the dataset images and split them into two lists - cars and notcars.  

Then I extracted the features using multiple_img_features functions described above. (Details of each of the features is given in the previous section)
This gave me a combined list of all the features of all the images which can not be used for training the classifier.

I used StandardScaler() to normalize the data and defined the output categories using np.hstack. There were two output categories 1 'cars', 0 'notcars'.

Then I used "Linear SVC" and fit the data using the fit() method of SVC object.

The output of training classifier is below - 

```bash 
96.73 Seconds to extract HOG features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
107.43 Seconds to train SVC...
Test Accuracy of SVC =  0.9896
My SVC predicts:  [ 1.  1.  1.  1.  1.  0.  0.  1.  0.  1.]
For these 10 labels:  [ 1.  1.  1.  1.  1.  0.  0.  1.  0.  1.]
0.19301 Seconds to predict 10 labels with SVC
Saving data to pickle file...
Data cached in pickle file.
```

It identified all the 10 test images successfully. 


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this is under the pipeline function in main.py file. The other helper functions to extract windows are in common.py file.

I decided to use multiple search windows based on estimation of what the size of the car will be and where in the frame the object may appear. 

Window 1 - 75 x75
Window 2 - 100 x 100
Window 3 - 150 x 150

Following is the visualization of search windows and there positions in the frame -

![small window][image5]
![medium window][image6]
![large window][image7]

I tried various combinations of the window sizes on the test images. I also experimented changing the overlap.
Finally I fixed the above mentioned windows and overlap of 75% based on the results on test images and test video.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Final pipeline consists of the following phases - 

##### Read image using the cv2.imread
The function imread from cv2 library is consitent for JPGs and PNGs. Following the first image from the pipeline - 
![first image][image8]

##### Convert the image to the HLS 
HLS works better in light and dark patches as compared to raw RGB. Following is the image -
![hls image][image9]

##### Get the search windows from the image using sliding window -
![combined window image][image10]

##### Extract the features from individual windows
Details of features extracted is given the first and second sections.

##### Predict the window image 
svc.predict function was used to determine if the window contains car or not.

##### Add the heat for 10 consecutive frames and generate a heat map 
![heatmap image][image11]

##### Draw bounding boxes on the image 
![Final image][image12]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The method I used to filter the false positives is using the heatmap.
The function add_heat is modified such that it combines the heat from 10 frames and then return the combined threshold. 
This method filters out the false positives which comes for less than 10 frames.

Apart from this, overlapping bounding boxes were used to heat the area futher where the car is present.
A threshold of 5 was then decided which was giving satisfactory results.

The code is in common.py line number 209.

It is still not able to completely filter all the false positivies. I think adjusting the parameters in the first phase of the pipeline will help solve it.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problem I faced in the project are as following -

##### Even after using the heatmap there are false positives. This means the parameters are correct entirely.
Solution: I will have to understand HOG and other feature extraction method completely to tweak the parameters such that there are minimal false positives.

##### The pipeline runs very slow on live video.
Solution: If we extract the features once for the entire frame, it will speed up the pipeline. However, I am not fully understanding that method hence I used individual window and extracted features from it.

The pipeline will fail in following situtions - 
The car is getting occluded for few frames and then coming back as it needs 10 frames to track it down.
Difference light conditions from daylight

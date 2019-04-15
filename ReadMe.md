# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./New_Images/2.png "Traffic Sign 1"
[image5]: ./examples/3.png "Traffic Sign 2"
[image6]: ./examples/4.png "Traffic Sign 3"
[image7]: ./examples/5.png "Traffic Sign 4"
[image8]: ./examples/6.png "Traffic Sign 5"

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is  `41018`
* The size of the validation set is `10255`
* The size of test set is `12630`
* The shape of a traffic sign image is `32,32,3`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

See the result in `traffic_sign_classifier.ipynb` under the heading Exploratory data analysis

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried out different image preprocessing techniques which included grayscale conversion and normalisation. Out of the 2 techniques mentioned I  decided to proceed with only normalising the images in the range [0,1] by dividing each pixel intensity by 255. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I decided to generate additional data because many classes in the original training dataset had very few images. I choose the an average number of 809 (34799/43) images as the minimum requirement for each class

To add more data to the the data set, I used the following techniques:

* Pixel Intensity rescaling
* Image Rotation
* Image horizontal flip
* Image vertical flip
* Image horizontal flip
* Image blur

Example of an original image and an augmented images are given in the `augment_data.ipynb` notebook from cells 6 - 10:

![alt text

The original  training dataset had `34799` images across all the classes and the augmented data had `41018` images across all the classes.
I made use of `sklearn train test split` function to split the augemented dataset into training and validation sets.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the Lenet model architecture with the additon of droupout layers in the 3rd and 4th layers to avoid overfitting

My final model consisted of the following layers:

| Layer         1 - 5	|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| Max pooling			| 2x2 stride, same padding, outputs 14x14x16	|
| RELU					|												|
| Input         		| 14x14x16 image 							    | 
| Convolution 4x4     	| 1x1 stride, same padding, outputs 10x10x16 	|
| Max pooling			| 2x2 stride, same padding, outputs 5x5x16  	|
| RELU					|												|
| Input         		| 5x5x16  image   						    	| 
| Fully connected       | outputs 400                               	|
| RELU					|												|
| Drouput																|
| Input         		| 400 neurons            					    | 
| RELU					|												|
| Drouput																|
| Input                 | 120 neurons     				            	|
| Fully connected		|  ouput 84     								|
| Softmax				|            									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the 
* `AdamOptimizer` 
* `batch size =128`
* `learning rate= 0`
* `Epochs  = 50`
 
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of `95.1`
* test set accuracy of `93.6`

If an iterative approach was chosen:
* What was the first architecture that was tried was the original LeNet architecture because it performed good on the MNest dataset, also the LeNet architecture inherently accepts 32x32x3 RGB images which required very less preprocessing of the default images from my side.

* The original architecture gave a 90% accuracy on thte original dataset but after augmenting the data applying the same model caused a decrease in the accuracy due to overfitting and thus I choose to add drouput layers in layer 3 and 4 of the LeNet architure to avoid overfitting.

* Mostly the drouput probability was tuned from 0.5 to 0.75, the batch size, learning rate and the epochs were also changed in order to increase the accuracy and speed of the training process

* How does the final model's accuracy on the training was .97, validation accuray was .951 and testing accuracy was 0.936
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/hr      		| End of speed limit (80km/h)  					| 
| Road work  			| Road work 									|
| End of speed (30km/h) | End of speed limit (80km/h)					|
| Speed limit (60km/h) 	| Speed limit (60km/h)					 		|
|  General caution		|  General caution   							|


The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 156th cell of the Ipython notebook.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


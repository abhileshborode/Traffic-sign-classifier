# **Traffic Sign Recognition** 



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
[image4]: ./New_Images/2.ppm "Traffic Sign 1"
[image5]: ./examples/3.ppm "Traffic Sign 2"
[image6]: ./examples/4.ppm "Traffic Sign 3"
[image7]: ./examples/5.ppm "Traffic Sign 4"
[image8]: ./examples/6.ppm "Traffic Sign 5"

---



### Data Set Summary & Exploration


I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is  `41018`
* The size of the validation set is `10255`
* The size of test set is `12630`
* The shape of a traffic sign image is `32,32,3`
* The number of unique classes/labels in the data set is `43`

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

See the result in `traffic_sign_classifier.ipynb` under the heading Exploratory data analysis

### Design and Test a Model Architecture



I tried out different image preprocessing techniques which included grayscale conversion and normalisation. Out of the 2 techniques mentioned I  decided to proceed with only normalising the images in the range [0,1] by dividing each pixel intensity by 255. 



I decided to generate additional data because many classes in the original training dataset had very few images. I choose the an average number of 809 (34799/43) images as the minimum requirement for each class

To add more data to the the data set, I used the following techniques:

* Pixel Intensity rescaling
* Image Rotation
* Image horizontal flip
* Image vertical flip
* Image horizontal flip
* Image blur

Example of an original image and an augmented images are given in the `augment_data.ipynb` notebook from cells 6 - 10:



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
| Fully connected		|  output 84     								|
| Softmax				|            									|


#### 3. Training parameters . 
To train the model, I used the 
* `AdamOptimizer` 
* `batch size =128`
* `learning rate= 0`
* `Epochs  = 50`
 
#### 4. Approach

My final model results were:
* validation set accuracy of `95.1`
* test set accuracy of `93.6`

If an iterative approach was chosen:
* What was the first architecture that was tried was the original LeNet architecture because it performed good on the MNest dataset, also the LeNet architecture inherently accepts 32x32x3 RGB images which required very less preprocessing of the default images from my side.

* The original architecture gave a 90% accuracy on thte original dataset but after augmenting the data applying the same model caused a decrease in the accuracy due to overfitting and thus I choose to add drouput layers in layer 3 and 4 of the LeNet architure to avoid overfitting.

* Mostly the drouput probability was tuned from 0.5 to 0.75, the batch size, learning rate and the epochs were also changed in order to increase the accuracy and speed of the training process

* How does the final model's accuracy on the training was .97, validation accuray was .951 and testing accuracy was 0.936
 

### Test a Model on New Images


The five German traffic signs that I found on the web are in the New_Images folder

#### The model's predictions 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/hr      		| End of speed limit (80km/h)  					| 
| Road work  			| Road work 									|
| End of speed (30km/h) | End of speed limit (80km/h)					|
| Speed limit (60km/h) 	| Speed limit (60km/h)					 		|
|  General caution		|  General caution   							|


The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%.

####  Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 156th cell of the Ipython notebook.

## Dependencies

* `OpenCV`
* `Numpy` 
* `Movie.py`
* `sklearn`
* `Pandas`
* `Pickle`
* `Tensorflow`

## Run Instructions
```bash
jupyter notebook augment_data.ipynb
jupyter notebook traffic-sign-classifier.ipynb 
```








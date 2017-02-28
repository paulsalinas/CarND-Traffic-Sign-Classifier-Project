#**Traffic Sign Recognition** 

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

[image1]: ./write_up_images/number_of_test_samples_vis.png "Visualization"
[image1a]: ./write_up_images/training_set_class_id_dist_vis.png "Sample Distribution"
[image2]: ./write_up_images/gray_scale.png "Grayscaling"
[image4]: ./write_up_images/20km_hr.jpg "20 km hr"
[image5]: ./write_up_images/main_street_right_of_way.jpg "main street right of way"
[image6]: ./write_up_images/no_entry.jpg "no entry"
[image7]: ./write_up_images/priority_road.jpg "Traffic Sign 4"
[image8]: ./write_up_images/stop.jpg "stop"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the python notebook cell marked as '2'. 

I used numpy to calculate the following stats:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in python notebook cell marked as '3'. 

Here is a bar chart that illustrates the number of training data vs. validation vs. test.

![alt text][image1]

Here is a bar chart that illustrates the sample distribution in our training set. As you can see this is a problem as the distribution is not even.

![alt text][image1a]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the python notebook's cell marked as '8'

As a first step, I decided to convert the images to grayscale. I did this because I found that my validation results improved when doing this vs. keeping RGB values in the input. This might be because the added color information might lead to incorrect training assumptions in our model as a result of poor lighting or photo quality. I found that this improved the model by 2-3% from the one that included color.

As a last step, I used a histogram normalization to accentuate the contrast between details. This is important because some sample images have blurred or dim details/features which would be difficult for our model to notice. I found that adding this improved my results against the validation set by an additional 2-3% than without this normalization step.

Here is an example of a traffic sign after grayscaling and applying histogram normalization.

![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Here is a bar chart that illustrates the number of training data vs. validation vs. test.

![alt text][image1].

For the time being, I decided against augmenting the data set. I do realize that this may improve my model as I can use different orientations and distortions to make our model learn these features under different circumstances. It could also help even out our training set sample distribution across class ids.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located python notebook's cell marked '14'.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scaled image   				    | 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 5x5x16 	|
| RELU					|												|
| DROPOUT				| 0.5 keep probability							|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| Flatten       	    | output 400   									|
| Fully Connected		| output 120    								|
| RELU					|												|
| Fully Connected		| output 84    		    						|
| RELU					|												|
| Fully Connected		| output 43    		    						|
| Softmax				|           									|

As you can see it uses the core LeCun network given in the lab with the exception of the dropout layer in between the second conv layer and pooling. This is to prevent overfiitting and I found that this improved my results against validation slightly.

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the python notebook's 16-18 code cell.

To train the model, I used 100 epochs and a batch size of 128. For the optimizer, I used the Adam optimizer used in the LeCun lab.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for training the model is located in the python notebook's 18-19 code cell.

My final model results were:
* validation set accuracy of 0.969 
* test set accuracy of 0.942

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* Answer: I used the LeCun architecture from the lab as it provided a good starting point to build from. The Lecun architecture used two layers of convolution to allow for feature detection across an image and fits well for this particular problem.

* What were some problems with the initial architecture?
* Answer: The initial results gave roughly 88% validation result which was not high enough and didn't have any regularization to prevent over fitting.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Answer: A dropout layer was added between the second convolution layer and pooling to prevent over fitting.

* Which parameters were tuned? How were they adjusted and why?
* Answer: I played around with a few parameters but nothing led to substantial improvement in the validation result. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
* Answer: A convolution layer fits this problem perfectly because it allows for the model to detect features independent of where they are in the photo. A dropout layer in this case will help with overfitting as it will prevent the model from relying on certain weights to make its prediction


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

for the 4th image the quality of the image might be a problem as it's a bit more pixelated than other images.

For the 5th image, there are some cropping on the sides and this might be a problem if the the model depends on those edges in order to predict this particular classification.

I think the rest are pretty straightforward and I'm not sure if I can identify any particular thing that might effect classification for these particular images.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is in the python notebook's 62nd cell.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)	| round about mandatory                     	|
| Right-of-way next intersection|  Right-of-way next intersection    	| 
| no-entry     			| turn-left ahead 								|
| Priority Road			| Priority Road									|
| Stop		        	| Stop      			                		|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is in the python notebook's 62nd cell.

#####image: Speed limit (20km/h) 
| Classification Id         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|40 (round about mandatory) | 0.592226|
|8| 0.168671|
|0| 0.0565114|
|5| 0.0526706|
|1| 0.0502788|


Here the above classifier had trouble effectively classifying the speed limit sign as it was incorrectly classified to a 'round about mandatory' sign. It's possible that there's some overfitting here that's causing this error especially to a classification (class id: 40) that did have as many training samples as the rest of the classifications.

#####image: Right-of-way next intersection
| Classification Id         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|11 (Right-of-way next intersection) | 0.999568|
|30| 0.000432218|
|23| 1.64366e-07|
|24| 1.71918e-08|
|29| 1.5021e-08|

High confidence here :)

#####image: no-entry
| Classification Id         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|34 (turn-left-ahead) | 0.555983|
|9| 0.425989|
|33| 0.00848603|
|17| 0.00517172|
|15| 0.00104859|

Here the classifier had trouble effectively classifying the no-entry sign and mistakenly classified it as a 'turn-left-ahead' sign. Again, there might be some overfitting here for a classification (class id: 34) that did not have many training samples compared to the rest.

#####image: priority road
| Classification Id         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|12 (priority-road)| 0.999903
|2| 9.08012e-05
|3| 3.15513e-06
|15| 8.27905e-07
|9| 6.39524e-07

High confidence here :)

#####image: Stop
| Classification Id         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|14 (stop)| 0.99913|
|25| 0.000234099|
|5| 0.000159833|
|13| 0.000115817|
|38| 0.000113825| 

High confidence here :)
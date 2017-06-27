#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./fig/architecture.png "Network Architecture"
[image2]: ./augment/origin_img.jpg ""Original image"
[image3]: ./augment/crop_img.jpg "Cropping"
[image4]: ./augment/bright_img.jpg "Random brightness"
[image5]: ./augment/flip_img.jpg "Flipping"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

For an appropriate model architecture, I reviewed three models (NVIDIA model, nconda's model, and naokishibuya) which has been evaluated. These models has own characteristics. Especially, the nconda's model has a strong advantage to use training dataset. This model used just one dataset received from udacity. For this reason, I chose it as my basement model. However, other models also were applied to implement my behavioral cloning algorithm.

The neural network architecture has nine layers: a normalization layer, five convolution layers, and three fully-connected layers. The detailed is described below. 

![alt text][image1]



This architecture and network image came from [NVIDIA's End-to-End Deep Learning](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).



####2. Attempts to reduce overfitting in the model

To prevent overfitting, I added dropouts between layers.  Each dropout has the same probability 0.3. Actually, these values were chosen by try & err method.

Furthermore, I shuffled and split the whole training data into X_train and y_valid to reduce overfitting using `sklearn.model_selection.train_test_split()`. The test_size is 0.1 and ramdon_state is 14 in `model.py`

####3. Model parameter tuning

In this model, I used `adam` optimizer. In addition, this model was trained based on some parameters as following:

samples_per_epoch = 24000

nb_epoch = 12

nb_val_samples=1024



####4. Appropriate training data

In the first try, I used only the sample training dataset provided by udacity. However, the vehicle frequently fell off the test track. So, I decided to gather additional training dataset using simulator. The strategy for collecting data was based on the guideline of udacity lecture. For more detailed strategy, see section 3. (Creation of the Training Set & Training Process)



###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was from the udacity lecture and the nividia end-to-end solution. The overall architecture was based on the nvidia's one.

I trained this model by using the sample training data provided by udacity. And run it in simulator. However, there were some spot where the vehicle fell off the track. To overcome this problem, I gathered more training dataset using simulator's manual mode.

After retraining, I succeeded to make the vehicle follow the track without leaving the road.

####2. Final Model Architecture

##### Preprocessing

To reduce the sky and the hood of car, I cropped image using `crop_image()`.  This method helped my model focus on an area of interest that consist of road and road boundary.  The origin image was cropped between 40 and 140 like red box in the following figure. (The left one is the original image and the other one is the cropped image from the origin. The red box refers to an area of interest)

![original image][image2]    ![cropped image][image3]



##### Network architecture

The final model architecture is 

*   Image normalization
*   Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
*   Drop out (0.3)
*   Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
*   Drop out (0.3)
*   Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
*   Drop out (0.3)
*   Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
*   Drop out (0.3)
*   Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
*   Drop out (0.3)
*   Fully connected: neurons: 100, activation: ELU
*   Drop out (0.3)
*   Fully connected: neurons: 50, activation: ELU
*   Drop out (0.3)
*   Fully connected: neurons: 10, activation: ELU
*   Drop out (0.3)
*   Fully connected: neurons: 1 (output)

Finally, this architecture has 252,219 parameters as following.

| Layer                           | Output shape       | Parameters | Connected to    |
| ------------------------------- | ------------------ | ---------- | --------------- |
| lambda_1(Lambda)                | (None, 66, 200, 3) | 0          | lambda_input_1  |
| convolution2d_1 (Convolution2D) | (None, 31, 98, 24) | 1824       | lambda_1        |
| dropout_1 (Dropout)             | (None, 31, 98, 24) | 0          | convolution2d_1 |
| convolution2d_2 (Convolution2D) | (None, 14, 47, 36) | 21636      | dropout_1       |
| dropout_2 (Dropout)             | (None, 14, 47, 36) | 0          | convolution2d_2 |
| convolution2d_3 (Convolution2D) | (None, 5, 22, 48)  | 43248      | dropout_2       |
| dropout_3 (Dropout)             | (None, 5, 22, 48)  | 0          | convolution2d_3 |
| convolution2d_4 (Convolution2D) | (None, 3, 20, 64)  | 27712      | dropout_3       |
| dropout_4 (Dropout)             | (None, 3, 20, 64)  | 0          | convolution2d_4 |
| convolution2d_5 (Convolution2D) | (None, 1, 18, 64)  | 36928      | dropout_4       |
| flatten_1 (Flatten)             | (None, 1152)       | 0          | convolution2d_5 |
| dropout_5 (Dropout)             | (None, 1152)       | 0          | flatten_1       |
| dense_1 (Dense)                 | (None, 100)        | 115300     | dropout_5       |
| dropout_6 (Dropout)             | (None, 100)        | 0          | dense_1         |
| dense_2 (Dense)                 | (None, 50)         | 5050       | dropout_6       |
| dropout_7 (Dropout)             | (None, 50)         | 0          | dense_2         |
| dense_3 (Dense)                 | (None, 10)         | 510        | dropout_7       |
| dropout_8 (Dropout)             | (None, 10)         | 0          | dense_3         |
| dense_4 (Dense)                 | (None, 1)          | 11         | dropout_8       |
|                                 | Total parms        | 252219     |                 |



####3. Creation of the Training Set & Training Process

##### Data collection

To prevent the vehicle to fall out the track, I should gather additional training dataset unlike the nconda's model. I followed udacity's tips for data collection:

-   four laps of center lane driving (2 clockwise / 2 counter-clockwise)
-   One lap of recovery driving from the sides
-   One lap focusing on driving smoothly around curves.

After retraining, this model for behavioral cloning was enhanced to follow the test track without leaving the road.



##### Image augmentation

I increase database by the image augmentation techniques. The first method is random brightness. In this model, I converted the brightness of the traning image randomly. The left(bottom) image is the original image, and the right one is the converted image whose brightness was dark. Through the method, i expected this model to be more general regardless brightness. 

![cropped image][image3]    ![random brightness][image4]

The other method for the augment is image flipping. The image (left) is mirrored at the Y axis to the flipped image (right). This method help to make a general model to directions by avoiding bias directions of steering angles. And the steering angle also should be changed reversely. In this example, the original steering angle (-0.2216 deg) was converted into 0.2216 deg

![cropped image][image3]     ![flipping][image5]



####4. Validation (the final video)

The final result of this model was uploaded on [YouTube](https://www.youtube.com/watch?v=l_UBaZDKgQU&feature=youtu.be)
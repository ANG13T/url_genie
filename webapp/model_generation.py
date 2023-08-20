'''
Malicious Domain Detection Model Generation Python Script
Created by Angelina Tsuboi (angelinatsuboi.com)

Objective:
This script was created with the goal of making a Multilayer Perceptron Neural Network

The code sequence is as follows:
1. Integrate CSV Dataset and Remove Unecessary Columns
2. Use SMOTE to Balance out Class Distribution in Dataset
3. Split Dataset into Training and Testing Sets using 80:20 Ratio
4. Initialize Multilayer Perception
5. Utilize Adam Optimization and Binary Cross Entropy Loss Function
6. Initialize Model Callback to Wait Until 0.1 Validation Loss 
7. Train Model with 10 Epochs and Batch Size of 256
8. Verify Model Results using 10 Examples
9. Save the Model into a .h5 File Output
'''
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Configuring Dataset and Values

# Reading in dataset from CSV file. This dataset is an updated version of the original Kaggle dataset including 
urldata = pd.read_csv("./Url_Processed.csv")

# Clean up dataset and remove unecessary columns
urldata.drop("Unnamed: 0",axis=1,inplace=True)
urldata.drop(["url","label"],axis=1,inplace=True)

# Configure dependent variables (values used to inform prediction)
x = urldata[['hostname_length',
       'path_length', 'fd_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]

# Configure independent variable (value to verify prediction)
y = urldata['result']

# Using SMOTE to resample dataset. The SMOTE (Synthetic Minority Over-sampling Technique) method is used to oversample the dataset
# SMOTE is used to balance the class distribution whenever it detects for an imbalance (one sample has signifcantly more samples than the other decreasing model performance) 
'''
Easy to understand example of SMOTE. Consider the following dataset with two features (X1 and X2) and a binary class label (y),
|  X1  |  X2  |  y  |
|------|------|-----|
|  1.5 |  2.0 |  0  |
|  2.0 |  3.0 |  0  |
|  3.0 |  5.0 |  1  |
|  3.5 |  4.5 |  0  |
|  4.0 |  3.5 |  0  |
|  4.5 |  4.0 |  0  |
|  5.0 |  2.5 |  1  |

There is an imbalance in the y column of the dataset as the class 1 is underrepresented.
If SMOTE is applied, the following output will be the result:
|  X1  |  X2  |  y  |
|------|------|-----|
|  1.5 |  2.0 |  0  |
|  2.0 |  3.0 |  0  |
|  3.0 |  5.0 |  1  |
|  3.5 |  4.5 |  0  |
|  4.0 |  3.5 |  0  |
|  4.5 |  4.0 |  0  |
|  5.0 |  2.5 |  1  |
| 2.75 | 4.25 |  1  |  # Synthetic sample (SMOTE)
| 4.25 | 2.75 |  1  |  # Synthetic sample (SMOTE)

As you can see, it generated two synthetic samples with the class 1 to balance out the class distribution in the dataset
'''
x_sample, y_sample = SMOTE().fit_resample(x, y.values.ravel())

x_sample = pd.DataFrame(x_sample)
y_sample = pd.DataFrame(y_sample)

# Seperate data into training and testing sets using the 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size = 0.2)


# Model Creation using Deep Learning (Multilayer Perceptron)
# The following lines of code are an implementation of a Multilayer Perceptron NN model using the Keras library
'''
A multilayer perception is a feedforward artificial neural network that consists of multiple layers of interconnected nodes also known as neurons. 
It is characterized by several layers of input nodes connected as a directed graph between the input and output layers.
It also utilizes backpropagation for training the model.

Input Layer (16 features)
     ↓
  Hidden Layer (32 neurons, ReLU)
     ↓
  Hidden Layer (16 neurons, ReLU)
     ↓
  Hidden Layer (8 neurons, ReLU)
     ↓
Output Layer (1 neuron, Sigmoid)
'''
# we create a sequential model (linear stack of layers where you can add successive layers with inputs and outputs)
model = Sequential()
# first layer of the model. It is a dense layer (fully connected layer) with 32 neurons. It utilizes ReLU (Rectified Linear Activation) which introduces non-linearity and takes in 16 input features
model.add(Dense(32, activation = 'relu', input_shape = (16, )))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
# the final layer is an output layer with one neuron which is utilized for binary classification with sigmoid classification that outputs a probability score between 0 and 1 (0 = no probability and 1 = full chance).
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Define an Optimizer
# the following line defines an Adam Optimization algorithm with a learning rate of 0.0001 which defines the step size during optimization of a model's parameters such as weights and biases
opt = keras.optimizers.Adam(lr=0.0001)
# the below line compiles the NN model with the following configurations:
# 1. Specifies a Binary Cross Entropy Loss Function which the model will minimize during training. It is the difference between the predicted probabilities and actual binary labels
# 2. Establishes the accuracy metric which is the metric evaluated and reported during training
model.compile(optimizer= opt ,loss='binary_crossentropy',metrics=['acc'])

# Define Callback Function
# The following code defines a callback function executed at the end of each epoch during the training of the model
class ModelCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # checks if the validation loss is less than 0.1
        if(logs.get('val_loss')<0.1):
            print("\nReached 0.1 val_loss! Halting training!")
            self.model.stop_training = True

callback = ModelCallback()

# Trains the model using the following parameters
# 1. 10 Epochs (number of times the training data should be iterated during training)
# 2. 256 Batch Size (the number of samples used in each epoch for updating model parameters). Batches optimize memory training and speed training
# 3. Verbosity of 1 (progress info will be displayed after each epoch which contains loss and accuracy data)
history = model.fit(x_train, y_train, epochs=10,batch_size=256, callbacks=[callback],validation_data=(x_test,y_test),verbose=1)

# list all data in history
print(history.history.keys())

# TEST SUITE
pred_test = model.predict(x_test)
for i in range (len(pred_test)):
    if (pred_test[i] < 0.5):
        pred_test[i] = 0
    else:
        pred_test[i] = 1
pred_test = pred_test.astype(int)

def view_result(array):
    array = np.array(array)
    for i in range(len(array)):
        if array[i] == 0:
            print("Safe")
        else:
            print("Malicious")

print("PREDICTED RESULTS: ")
view_result(pred_test[:10])
print("\n")
print("ACTUAL RESULTS: ")
view_result(y_test[:10])

# SAVE MODEL
model.save("Malicious_URL_Prediction.h5")

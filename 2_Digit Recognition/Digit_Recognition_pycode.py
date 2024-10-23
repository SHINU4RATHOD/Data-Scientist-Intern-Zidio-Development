'''
1. Business understanding:
    Business Problem:
    Manual Processing of Handwritten Documents:-
    Many industries, such as banking (for processing checks), postal services (for reading postal codes),
    and retail (for reading receipts), rely heavily on manual data entry when it comes to processing handwritten digits. 
    This manual process is slow, prone to errors, and inefficient, leading to:

    High labor costs.
    Delays in processing time.
    Errors in data interpretation and entry.
    Limited scalability for high-volume processing.


Business Solution:
Implementing a digit recognition system using deep learning (CNNs) automates the reading and digitization 
of handwritten inputs, increasing accuracy, reducing human error, and speeding up processes, ultimately 
saving time and costs for businesses.


2. Data Collection: 
    MNIST dataset handwritten Black & White images of digits with 28*28 pixels.
    
    
    Digit recognition, often referred to as handwritten digit recognition, is a popular machine learning 
    task where the goal is to classify images of handwritten digits into one of the ten classes (0 through 9).
    A well-known dataset for this task is the MNIST dataset, which contains 70,000 grayscale images of handwritten 
    digits, each of size 28x28 pixels.
'''


# Importing necessary libraries for data manipulation, visualization, and deep learning model development
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sqlalchemy import create_engine
# Load training and testing datasets from local machine
train_df = pd.read_csv(r"C:/Users/SHINU RATHOD/Desktop/Class/ML/26_DL & Ai part/01_ANN/study material/Neural Network-ANN/Neural Network-ANN/train_sample.csv")
test_df = pd.read_csv(r"C:/Users/SHINU RATHOD/Desktop/Class/ML/26_DL & Ai part/01_ANN/study material/Neural Network-ANN/Neural Network-ANN/test_sample.csv")

# Database connection for loading the training and testing datasets into MySQL for storage and future queries
user = 'root'  # Database username
pw = '1122'  # Database password
db = 'zidio_development'  # Database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
# train_df.to_sql('Digit_Recogn_train_df', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
# test_df.to_sql('Digit_Recogn_test_df', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# Retriving train and test data from the SQL database
sql1 = 'select * from Digit_Recogn_train_df;'
train = pd.read_sql_query(sql1, engine)

sql2 = 'select * from Digit_Recogn_train_df;'
test = pd.read_sql_query(sql2, engine)


# Visualizing the distribution of labels in the training dataset to check for class imbalance
sns.countplot(x='label', data=train_df)  # Dataset is balanced

# Displaying a sample image from the dataset to verify data correctness
i = 54
img = train_df.iloc[i, 1:].values.reshape((28, 28))
plt.imshow(img, cmap="gray")
plt.title(train_df.loc[i, 'label'])
plt.show()


# Converting input images from integer type to float32 for compatibility with neural network training
x_train = train.iloc[:, 1:].values.astype("float32")  # Features (image pixel values)
x_test = test.iloc[:, 1:].values.astype("float32")  # Test features
y_train = train.label.values.astype("float32")  # Labels (digit classes)
y_test = test.label.values.astype("float32")  # Test labels

# Normalizing pixel values to fall within the range [0, 1] for better performance of the neural network
x_train = x_train / 255
x_test = x_test / 255

# Reshaping data to be compatible with CNN input (28x28x1 for grayscale images)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# One-hot encoding of the target labels (digits 0-9) for multi-class classification
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_of_classes = y_test.shape[1]
# Checking the shape of the processed data to ensure everything is correct before feeding into the model
x_train.shape
y_train.shape
x_test.shape
y_test.shape
 

#For displaying an image
plt.imshow(x_train[15])
plt.show
print(y_train[15])


# CNN Model - Defining the architecture for digit recognition using Convolutional Neural Network (CNN)
# Initialize a Sequential model
model = models.Sequential()
# Adding convolutional and pooling layers to extract features from the images
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # First convolutional layer with 32 filters
model.add(layers.MaxPooling2D((2, 2)))  # First max pooling layer to reduce dimensions
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Second convolutional layer with 64 filters
model.add(layers.MaxPooling2D((2, 2)))  # Second max pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Third convolutional layer


# Adding fully connected layers to perform classification after the feature extraction
model.add(layers.Flatten()) # # Flattening the 2D output from convolution layers into 1D

# Add a Dense layer for classification
model.add(layers.Dense(64, activation='relu'))  # Fully connected layer
model.add(layers.Dense(10, activation='softmax'))  # Output layer with softmax for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the updated model summary
model.summary()

# OPTIONAL: Data Augmentation to improve model generalization by creating variations of the images
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,   # Rotate images up to 10 degrees
    width_shift_range=0.1,  # Shift images horizontally by 10%
    height_shift_range=0.1,  # Shift images vertically by 10%
    zoom_range=0.1  # Zoom images by 10%
)

# Fit the data generator on the training data
# datagen.fit(x_train)
# Train the model with augmented data
# history = model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=20, validation_data=(x_test, y_test))

# Train the model without augmented data
history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))


# Plotting the training and validation accuracy across epochs
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting the training and validation loss across epochs
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Saving the trained model architecture and weights for future inference
model_json = model.to_json()  # Serializing model to JSON format
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")  # Saving the model's weights in HDF5 format


#################################################################################
#### Testing on new sample data
# LOADING AND TESTING SAVED MODEL
from tensorflow.keras.models import model_from_json

# Loading model from saved JSON and HDF5 files
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.weights.h5")  # Loading model weights
loaded_model.summary()  # Check if the last layer has the output shape of (None, 10)

# Compiling the loaded model for evaluation
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Loading the test dataset for making predictions
test = pd.read_csv(r"C:/Users/SHINU RATHOD/Desktop/Class/ML/26_DL & Ai part/01_ANN/study material/Neural Network-ANN/Neural Network-ANN/test.csv")
print(test.shape)

test_pred = test.values.astype('float32') / 255  # Normalize the pixel values
test_pred = test_pred.reshape((test_pred.shape[0], 28, 28, 1))  # Reshape to (num_samples, 28, 28, 1)

predictions = loaded_model.predict(test_pred)
print(predictions.shape)  # This should be (num_samples, 10), where 10 is the number of classes


# Now apply np.argmax to get the predicted labels
predicted_labels = np.argmax(predictions, axis=1)  # Choose the class with highest probability
predicted_labels

# Convert to DataFrame to match the desired format
result = pd.DataFrame(predicted_labels, columns=['Label'])
result
#################################################################################

 
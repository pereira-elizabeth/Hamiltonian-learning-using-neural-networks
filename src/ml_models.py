import tensorflow as tf

# Keras for neural networks
from keras.models import load_model
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from tensorflow.keras import layers, models, optimizers

# Create neural network model
def create_model(input_shape):  #the neural network that predicts v1 and v2 from the corresponding spectral density
    x_input = Input(shape=input_shape, name="x_input")
    x = Dense(300, activation="relu", kernel_initializer="glorot_uniform")(x_input)
    x = Dense(100, activation="relu", kernel_initializer="glorot_uniform")(x)
    x = Dense(50, activation="relu", kernel_initializer="glorot_uniform")(x)
    output = Dense(1)(x)
    model = Model(x_input, output)
    return model

#We see a slight overfitting of the neural network on the training data in the presence of noise, we improve this by 
#using a neural network that has more dropout layers and different learning rates

def create_model_prevent_overfitting(input_shape): #model to prevent overfitting
    # Input layer
    x_input = Input(shape=input_shape, name="x_input")
    # First Dense layer with L2 regularization and Dropout
    x = Dense(300, activation="relu", kernel_initializer="glorot_uniform", 
              kernel_regularizer=l2(0.001))(x_input)
    x = Dropout(0.3)(x)  # Dropout with a rate of 0.3
    # Second Dense layer with L2 regularization and Dropout
    x = Dense(100, activation="relu", kernel_initializer="glorot_uniform", 
              kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)  # Dropout with a rate of 0.3
    # Third Dense layer with L2 regularization and Dropout
    x = Dense(50, activation="relu", kernel_initializer="glorot_uniform", 
              kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)  # Dropout with a rate of 0.3
    # Output layer
    output = Dense(1)(x)
    # Create and return the model
    model = Model(x_input, output)
    return model

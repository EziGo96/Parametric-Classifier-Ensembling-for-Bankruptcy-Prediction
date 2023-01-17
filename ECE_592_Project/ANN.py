'''
Created on 10-Nov-2021

@author: somsh
'''
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
class ANN:
    @staticmethod
    def build(width,height,classes,reg=0.002):
        inputShape = (width, height)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (height, width)
            chanDim = 1           
        inputs = Input(shape=inputShape)
        x = Flatten()(inputs)
        
        # x = Dense(96, kernel_regularizer=l2(reg))(x)
        # x = Activation("sigmoid")(x)
        # x = BatchNormalization(axis=chanDim)(x)
        # x = Dropout(0.5)(x)
        
        x = Dense(48, kernel_regularizer=l2(reg))(x)
        x = Activation("sigmoid")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)
        
        x = Dense(48, kernel_regularizer=l2(reg))(x)
        x = Activation("sigmoid")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)
        
        x = Dense(24, kernel_regularizer=l2(reg))(x)
        x = Activation("sigmoid")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)
        
        x = Dense(12, kernel_regularizer=l2(reg))(x)
        x = Activation("sigmoid")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)
        
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
        model = Model(inputs, x, name="ANN")
        return model
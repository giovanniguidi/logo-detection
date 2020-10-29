import os
import numpy as np
import cv2
import random
import datetime
import io
import json
#import keras
import string

from base.base_model import BaseModel

import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Conv2DTranspose, Add, Concatenate, Flatten
#from keras.callbacks import EarlyStopping
#import keras.backend as K
from tensorflow.keras.optimizers import Adam

#from keras.applications.vgg16 import VGG16
#from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
#from keras.models import model_from_json
from losses.custom_losses import custom_categorical_crossentropy

#from models.encoder import encoder_graph, encoder_graph_vgg16
#from models.decoder import decoder_graph_8x, decoder_graph_16x, decoder_graph_32x


class Network(BaseModel):
#def Network (config): #input1 : [Batch_size, 256, 256, 3], input2 : [Batch_size, 64, 64, 3]        
    def __init__(self, config):
        """
        Constructor
        """
        super().__init__(config)
        self.y_size = self.config['image']['image_size']['y_size']
        self.x_size = self.config['image']['image_size']['x_size']
        self.num_channels = self.config['image']['image_size']['num_channels']
        self.num_classes = 3
        self.use_pretrained_weights = self.config['train']['weights_initialization']['use_pretrained_weights']
        self.train_from_scratch = self.config['network']['train_from_scratch']
        self.graph_path = self.config['network']['graph_path']
        self.learning_rate = self.config['train']['learning_rate']
#        self.decoder = self.config['network']['decoder']
#        self.session = tf.Session()
#        self.input1 = tf.placeholder(dtype = tf.float32, shape = [batch_size, 256, 256, 3], name = 'Target')
#        self.input2 = tf.placeholder(dtype = tf.float32, shape = [batch_size, 64, 64, 3], name = 'Query')
        self.model = self.build_model()
#        self.save_graph( self.model, self.graph_path)

    def build_model(self):
        model = self.build_graph()        
#        model.compile(optimizer = "adam", loss = self.loss)
#        model.compile(optimizer = "adam", loss = "categorical_crossentropy")
#        model.compile(optimizer = "adam", loss = "binary_crossentropy")

        optimizer = Adam(learning_rate = self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

        model.compile(optimizer = optimizer, loss = custom_categorical_crossentropy())
        model.summary()
        return model


    def build_graph(self):
        input_graph_image = Input(shape=(256, 256, 3), name='input_image_1')

        input_graph_query = Input(shape=(64, 64, 3), name='input_image_2')


        conv1_1 = Conv2D(filters = 64, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="conv1_1")(input_graph_image)

        conv1_2 = Conv2D(filters = 64, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="conv1_2")(conv1_1)

        pool1 = MaxPooling2D(pool_size=2, padding='same', name='pool1')(conv1_2)      

        #-----
        conv2_1 = Conv2D(filters = 128, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="conv2_1")(pool1)

        conv2_2 = Conv2D(filters = 128, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="conv2_2")(conv2_1)

        pool2 = MaxPooling2D(pool_size=2, padding='same', name='pool2')(conv2_2)      

        #-----
        conv3_1 = Conv2D(filters = 256, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="conv3_1")(pool2)

        conv3_2 = Conv2D(filters = 256, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="conv3_2")(conv3_1)

        pool3 = MaxPooling2D(pool_size=2, padding='same', name='pool3')(conv3_2)      

        #-----
        conv4_1 = Conv2D(filters = 512, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="conv4_1")(pool3)

        conv4_2 = Conv2D(filters = 512, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="conv4_2")(conv4_1)

        pool4 = MaxPooling2D(pool_size=2, padding='same', name='pool4')(conv4_2)    


        #-----
        conv5_1 = Conv2D(filters = 512, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="conv5_1")(pool4)

        conv5_2 = Conv2D(filters = 512, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="conv5_2")(conv5_1)

        pool5 = MaxPooling2D(pool_size=2, padding='same', name='pool5')(conv5_2)    


        #----------conditional

        Bconv1_1 = Conv2D(filters = 32, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Bconv1_1")(input_graph_query)

        Bconv1_2 = Conv2D(filters = 32, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Bconv1_2")(Bconv1_1)

        Bpool1 = MaxPooling2D(pool_size=2, padding='same', name='Bpool1')(Bconv1_2)   


        #----------
        Bconv2_1 = Conv2D(filters = 64, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Bconv2_1")(Bpool1)
        Bconv2_2 = Conv2D(filters = 64, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Bconv2_2")(Bconv2_1)
        Bpool2 = MaxPooling2D(pool_size=2, padding='same', name='Bpool2')(Bconv2_2)   

        #----------
        Bconv3_1 = Conv2D(filters = 128, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Bconv3_1")(Bpool2)
#        Bconv3_2 = Conv2D(filters = 128, kernel_size = 3, padding='same',
#                    strides=1, activation='relu', use_bias=True, name="Bconv3_2")(Bconv3_1)
        Bpool3 = MaxPooling2D(pool_size=2, padding='same', name='Bpool3')(Bconv3_1)   


        #----------
        Bconv4_1 = Conv2D(filters = 256, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Bconv4_1")(Bpool3)
        Bpool4 = MaxPooling2D(pool_size=2, padding='same', name='Bpool4')(Bconv4_1)   

        #----------
        Bconv5_1 = Conv2D(filters = 512, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Bconv5_1")(Bpool4)
        Bpool5 = MaxPooling2D(pool_size=2, strides=2, padding='same', name='Bpool5')(Bconv5_1)   


        Bconv6 = Conv2D(filters = 512, kernel_size = 2, padding='valid',
                    strides=1, activation='relu', use_bias=True, name="Bconv6")(Bpool5)

        Btile1 = tf.tile(Bconv6, [1, pool5.get_shape().as_list()[1], pool5.get_shape().as_list()[2], 1])  #[32, 8, 8, 512]
        Btile2 = tf.tile(Bconv6, [1, conv5_2.get_shape().as_list()[1], conv5_2.get_shape().as_list()[2], 1]) #[32, 16, 16, 512]
        Btile3 = tf.tile(Bconv6,[1, conv4_2.get_shape().as_list()[1], conv4_2.get_shape().as_list()[2], 1])  #[32, 32, 32, 512]
        Btile4 = tf.tile(Bconv6,[1, conv3_2.get_shape().as_list()[1], conv3_2.get_shape().as_list()[2], 1])  #[32, 64, 64, 512]    
        Btile5 = tf.tile(Bconv6,[1, conv2_2.get_shape().as_list()[1], conv2_2.get_shape().as_list()[2], 1])  #[32, 128, 128, 512]

        #decoder

        efused_1 = Concatenate(axis=-1, name='efused_1')([pool5, Btile1])  #[32, 16, 16, 1024]

        #---------
        Dconv1_1 = Conv2D(filters = 512, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Dconv1_1")(efused_1)
        Dconv1_2 = Conv2D(filters = 512, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Dconv1_2")(Dconv1_1)

        upsam1 = Conv2DTranspose(512, 3, strides=(2, 2), padding='same', name='upsam1')(Dconv1_2)

        #-----
        efused_2 = Concatenate(axis=-1, name='efused_2')([conv5_2, Btile2])  #[32, 16, 16, 1024]

        #---------
        fconv_1 = Conv2D(filters = 512, kernel_size = 1, padding='same',
                    strides=1, activation='relu', use_bias=True, name="fconv_1")(efused_2)

        #-------
        dfused_2 = Concatenate(axis=-1, name='dfused_2')([upsam1, fconv_1])  
        #-----------
        Dconv2_1 = Conv2D(filters = 512, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Dconv2_1")(dfused_2)
        Dconv2_2 = Conv2D(filters = 512, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Dconv2_2")(Dconv2_1)

        upsam2 = Conv2DTranspose(512, 3, strides=(2, 2), padding='same', name='upsam2')(Dconv2_2)

        #------
        efused_3 = Concatenate(axis=-1, name='efused_3')([conv4_2, Btile3])  
        #------
        fconv_2 = Conv2D(filters = 256, kernel_size = 1, padding='same',
                    strides=1, activation='relu', use_bias=True, name="fconv_2")(efused_3)
        #---------
        dfused_3 = Concatenate(axis=-1, name='dfused_3')([upsam2, fconv_2])  
        #-------
        Dconv3_1 = Conv2D(filters = 256, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Dconv3_1")(dfused_3)
        Dconv3_2 = Conv2D(filters = 256, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Dconv3_2")(Dconv3_1)

        upsam3 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same', name='upsam3')(Dconv3_2)

        #------
        efused_4 = Concatenate(axis=-1, name='efused_4')([conv3_2, Btile4])  
        #------
        fconv_3 = Conv2D(filters = 128, kernel_size = 1, padding='same',
                    strides=1, activation='relu', use_bias=True, name="fconv_3")(efused_4)
        #---------
        dfused_4 = Concatenate(axis=-1, name='dfused_4')([upsam3, fconv_3])  
        #-------
        Dconv4_1 = Conv2D(filters = 128, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Dconv4_1")(dfused_4)
        Dconv4_2 = Conv2D(filters = 128, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Dconv4_2")(Dconv4_1)

        upsam4 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', name='upsam4')(Dconv4_2)

        #------
        efused_5 = Concatenate(axis=-1, name='efused_5')([conv2_2, Btile5])  
        #------
        fconv_4 = Conv2D(filters = 64, kernel_size = 1, padding='same',
                    strides=1, activation='relu', use_bias=True, name="fconv_4")(efused_5)
        #---------
        dfused_5 = Concatenate(axis=-1, name='dfused_5')([upsam4, fconv_4])  
        #-------
        Dconv5_1 = Conv2D(filters = 64, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Dconv5_1")(dfused_5)
        Dconv5_2 = Conv2D(filters = 64, kernel_size = 3, padding='same',
                    strides=1, activation='relu', use_bias=True, name="Dconv5_2")(Dconv5_1)

        upsam5 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', name='upsam5')(Dconv5_2)

        output = Conv2D(filters = 1, kernel_size = 3, padding='same',
                    strides=1, activation='sigmoid', use_bias=True, name="fconv_5")(upsam5)

#        output = upsam5

#        output = Reshape(output, (256, 256))
        output = Reshape(target_shape=(256, 256))(output)

        model = Model(inputs=[input_graph_query, input_graph_image], outputs=output)

        return model

    """
    def save_graph(self, model, graph_path):
        model_json = model.to_json()
        with open(graph_path, "w") as json_file:
            json_file.write(model_json)
    """


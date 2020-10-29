import os
import numpy as np
#import cv2
import io
import json
#import keras
import string

from base.base_data_generator import BaseDataGenerator
from data_generators.data_augmentation import data_aug_functions
from preprocessing.preproc_functions import read_image, normalize_0_mean_1_variance, normalize_0_1, read_annotation, read_query
#from keras.applications.vgg16 import preprocess_input
import random

#np.random.seed(34) 

class DataGenerator(BaseDataGenerator):
    def __init__(self, config, dataset, shuffle=True, use_data_augmentation=False):
        super().__init__(config, shuffle, use_data_augmentation) 
        self.dataset = dataset
        self.dataset_full_len, self.dataset_len =  self.get_dataset_len()
        self.full_indices = np.arange(self.dataset_full_len)
        self.indices = self.full_indices[0:self.dataset_len]
#        self.num_classes = self.config['network']['num_classes']
        self.on_epoch_end()

    def get_dataset_len(self):    
        dataset_full_len = len(self.dataset)

        if self.config['train']['train_on_subset']['enabled']:
            fraction = self.config['train']['train_on_subset']['dataset_fraction']
            dataset_len = int(dataset_full_len * fraction)
        else:
            dataset_len = dataset_full_len

        return dataset_full_len, dataset_len

    def __len__(self):

        return int(np.floor(self.dataset_len / self.batch_size))

    def __getitem__(self, index):
        
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        dataset_temp = [self.dataset[k] for k in indices]
        
        # Generate data
        X_query, X_image, y = self.data_generation(dataset_temp)
#        X_image, y = self.data_generation(dataset_temp)

        return [X_query, X_image], y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch 
        """
        if self.shuffle == True:
            np.random.shuffle(self.full_indices)
            self.indices = self.full_indices[0:self.dataset_len]

    def data_generation(self, dataset_temp):        
        
        batch_x_query = []
        batch_x_image = []
        batch_y = []

        for elem in dataset_temp:            
            y_size = self.config['image']['image_size']['y_size']
            x_size = self.config['image']['image_size']['x_size']
            y_query_size = self.config['image']['query_image_size']['y_size']
            x_query_size = self.config['image']['query_image_size']['x_size']

#            num_channels = self.config['image']['image_size']['num_channels']
#            convert_to_grayscale = self.config['image']['convert_to_grayscale']

            query = read_query(self.dataset_folder, elem['query'], y_query_size, x_query_size, black_white = False)
            image = read_image(self.dataset_folder, elem['image']['filename'], y_size, x_size, black_white = False)
            annotation = read_annotation(self.dataset_folder, elem['image']['annotation'], y_size, x_size)

#            print(elem['query']
#            print(query)
#            print(elem['image'])

            if elem['query']['class_num'] != elem['image']['class_num']:
#                print("different class")
                annotation[:,:] = 0
#            else:
#                print("same class")

#            print( elem['image']['annotation'])
#            print(np.max(annotation))
#            print(annotation.shape)
            
            if self.use_data_aug:
                #print('data aug')
                image, annotation = data_aug_functions(image, annotation, self.config)
            
#            annotation_one_hot = annotation
#            annotation_one_hot = convert_annotation_one_hot(annotation, y_size, x_size, num_classes = self.num_classes)
            query = normalize_0_1(query)                   
            image = normalize_0_1(image)                   
            annotation = normalize_0_1(annotation)                   

#            image = normalize_0_mean_1_variance(image)                   

            #print(image.shape)
            batch_x_query.append(query)
            batch_x_image.append(image)
            batch_y.append(annotation)
#            batch_y.append(annotation_one_hot)
        
        batch_x_query = np.asarray(batch_x_query, dtype = np.float32)
        batch_x_image = np.asarray(batch_x_image, dtype = np.float32)
        batch_y = np.asarray(batch_y, dtype = np.float32)
        
        return batch_x_query, batch_x_image, batch_y

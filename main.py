import argparse
import os
import numpy as np
import json
import yaml

from data_generators.data_generator import DataGenerator
from models.model import Network
from trainers.trainer import Trainer
from predictors.predictor import PredictorFCN
#from utils.score_prediction import score_prediction
from preprocessing.preproc_functions import read_image, read_query, normalize_0_1
#from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model, model_from_json
from skimage.measure import label, regionprops
#import cv2
from PIL import Image, ImageDraw

def train(args):
    """
    Train a model on the train set defined in labels.json
    """
    
    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    with open(config['labels_file']) as f:
        dataset = json.load(f)

#    print(dataset)
                
    train_generator = DataGenerator(config, dataset['train'], shuffle=True, 
                                    use_data_augmentation=config['data_aug']['use_data_aug'])
        
    #----------val generator--------
    val_generator = DataGenerator(config, dataset['val'], shuffle=True, use_data_augmentation=False)

    train_model = Network(config)
    trainer = Trainer(config, train_model, train_generator, val_generator)
    trainer.train()

    
def predict_on_test(args):
    """
    Predict on the test set defined in labels.json
    """
        
    config_path = args.conf
    
    with open(config_path) as f:
        config = yaml.load(f)
        
    with open(config['labels_file']) as f:
        dataset = json.load(f)
        
    test_generator = DataGenerator(config, dataset['train'], shuffle=False, use_data_augmentation=False)
    
    #numpy array containing images
    images_test, labels_test = test_generator.get_full_dataset()

    #print(images_test.shape)
    #print(len(labels_test))
    
#    graph_file =  config['network']['graph_path']
#    weights_file = config['predict']['weights_file']
#    batch_size = config['predict']['batch_size']
    
    predictor = PredictorFCN(config)
   
    pred_test = predictor.predict(images_test)

    pixel_accuracy, mean_accuracy, mean_IoU, freq_weighted_mean_IoU = score_prediction(labels_test, pred_test, 80)
    
#    for i in range(20):
#        print(labels_test[i], pred_test[i])
    
    print("pixel accuracy:", round(pixel_accuracy, 2))
    print("mean accuracy:", round(mean_accuracy, 2))
    print("mean IoU:", round(mean_IoU, 2))
    print("freq_weighted_mean_IoU:", round(freq_weighted_mean_IoU, 2))
    

def predict(args):
    """
    Predict on a single image 
    """
        
    config_path = args.conf

    query_filename =  args.query
    image_filename = args.image
    output_filename = args.output

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    y_size = config['image']['image_size']['y_size']
    x_size = config['image']['image_size']['x_size']
    y_query_size = config['image']['query_image_size']['y_size']
    x_query_size = config['image']['query_image_size']['x_size']


    query_dict = {"filename": query_filename}
    
    query = read_query("./", query_dict, y_query_size, x_query_size, black_white = False)
    image = read_image("./", image_filename, y_size, x_size, black_white = False)

    query = normalize_0_1(query)                   
    image = normalize_0_1(image) 
    query = np.expand_dims(query, axis=0)
    image = np.expand_dims(image, axis=0)

    #load model
    json_file = open(config['network']['graph_path'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(config['predict']['weights_file'])


    threshold = 0.5 #threshold for the prediction

    output_prob = model.predict([query, image])
    output = np.where(output_prob > threshold, 1., 0.)

    query = query[0, ...]
    image = image[0, ...]
    output = output[0, ...]

    lbl_0 = label(output) 
    props = regionprops(lbl_0)

    image_with_bbox = Image.fromarray(np.uint8(image * 255.))
    draw = ImageDraw.Draw(image_with_bbox)

    for prop in props:
        draw.rectangle(((prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2])), outline="red", width=3)

    image_with_bbox.save(output_filename)

#    image_with_bbox = np.asarray(image_with_bbox)/255.

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Seq2seq')
    parser.add_argument('-c', '--conf', help='path to configuration file', required=True)
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Train')    
    group.add_argument('--predict_on_test', action='store_true', help='Predict on test set')
    group.add_argument('--predict', action='store_true', help='Predict on single file')

    parser.add_argument('--query', help='path to query file')
    parser.add_argument('--image', help='path to image file')
    parser.add_argument('--output', help='path to output file')
    
    args = parser.parse_args()
   
    #    print(args)
      
    if args.predict_on_test:
        print('Predicting on test set')
        predict_on_test(args)      

    elif args.predict:
        if args.image is None:
            raise Exception('missing --image image_path')
        else:
            print('predict')
        predict(args)

    elif args.train:
        print('Starting training')
        train(args)       
    else:
        raise Exception('Unknown args') 

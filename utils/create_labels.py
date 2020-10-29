import json
from sklearn.model_selection import train_test_split
import os
from scipy.io import loadmat
import random
import numpy as np

def create_labels_flicklogos2(base_folder, area_threshold):

    #flickrLogos_32
    dataset_images = []
    dataset_query = []
#    areas = []

    dataset_folder = "./FlickrLogos-v2/classes/"
    data_folder = base_folder + dataset_folder
 
    logo_classes = os.listdir(data_folder + "jpg/")

    logo_classes.remove('no-logo')

    logo_classes = sorted(logo_classes)
    labels_dict = {value: key+1 for (key, value) in enumerate(logo_classes)}    

    print(labels_dict)

    for logo_class in logo_classes:
#        print(logo_class)

        images = os.listdir(data_folder + "jpg/" + logo_class)

#        print(images)

        for image in images:
            out_dict_images = {}

            #images                
            out_dict_images['filename'] = dataset_folder + "jpg/" + logo_class + "/" + image
            out_dict_images['annotation'] = dataset_folder + "masks/" + logo_class + "/" + image + ".mask.merged.png"
            out_dict_images['class_name'] = logo_class
            out_dict_images['class_num'] = labels_dict[logo_class]

            dataset_images.append(out_dict_images)

            #query
            out_dict_query = {}

            with open(data_folder + "masks/" + logo_class + "/" + image + ".bboxes.txt" ,"r+") as f:
                lines = f.readlines()
            lines = lines[1:]
            
            for line in lines:
                bbox = line.strip().split(" ")

                area = int(bbox[2]) * int(bbox[3])

                if area > area_threshold:
                    out_dict_query['filename'] = dataset_folder + "jpg/" + logo_class + "/" + image
                    out_dict_query['bbox'] = bbox
                    out_dict_query['class_name'] = logo_class
                    out_dict_query['class_num'] = labels_dict[logo_class]

                    dataset_query.append(out_dict_query)
#                    areas.append(area)

    return dataset_images, dataset_query, logo_classes, labels_dict

def create_labels_flicklogos47(base_folder, logo_classes, labels_dict):

    dataset_images = []
    dataset_query = []

    #flickrLogos_47
    dataset_folder = "./FlickrLogos_47/"
    data_folder = base_folder + dataset_folder

    with open(data_folder + "className2ClassID.txt","r+") as f:
        lines = f.readlines()

    dict_labels_FL47 = {}

    for line in lines:
        line = line.strip()
        line = line.split("\t")

        label_val = line[0].replace("_symbol", "")#.replace("_text", "")

        if label_val in logo_classes:
#        print(line)
            dict_labels_FL47[line[1]] = label_val

    print(dict_labels_FL47)

    splits = ['train', 'test']
    folders = ['000000', '000001', '000002']

    for split in splits:
        for folder in folders:
            filenames = os.listdir(data_folder + "/" + split + "/" + folder)
    #        print(filenames)

            for filename in filenames:
                if filename.endswith(".txt"):

                    with open( data_folder + "/" + split + "/" + folder + "/" + filename ,"r+") as f:
                        lines = f.readlines()

                    for line in lines:
                        line = line.strip().split(" ")
                        num_class = line[4]
                        mask_ext = line[6]
                        bbox = line[0:4]

                        if num_class in dict_labels_FL47.keys():
                            out_dict_images = {}
                            base_filename = filename.split(".")[0]

                            out_dict_images['filename'] = dataset_folder + split + "/" + folder + "/" + base_filename + ".png"
                            out_dict_images['annotation'] = dataset_folder + split + "/" + folder + "/" + base_filename + "." + mask_ext + ".png"
                            out_dict_images['class_name'] = dict_labels_FL47[num_class]
                            out_dict_images['class_num'] = labels_dict[out_dict_images['class_name']]

 #                           print(out_dict)
                            #dataset_images.append(out_dict_images)

                            #query
                            out_dict_query = {}
                            out_dict_query['filename'] = dataset_folder + split + "/" + folder + "/" + base_filename + ".png"
                            out_dict_query['bbox'] = bbox
                            out_dict_query['class_name'] = dict_labels_FL47[num_class]
                            out_dict_query['class_num'] = labels_dict[out_dict_images['class_name']]

                            #dataset_query.append(out_dict_query)

    print("len dataset query:", len(dataset_query))
    print("len dataset images:", len(dataset_images))

    return dataset_images, dataset_query

def create_labels(base_folder, val_size, test_size, random_state):

    dataset_images = []
    dataset_query = []

    area_threshold = 3000.

    dataset_images_flicklogos2, dataset_query_flicklogos2, logo_classes, labels_dict = create_labels_flicklogos2(base_folder, area_threshold)
    dataset_images += dataset_images_flicklogos2
    dataset_query += dataset_query_flicklogos2
#    dataset_images_flicklogos47, dataset_query_flicklogos47 = create_labels_flicklogos47(base_folder, logo_classes, labels_dict)
#    dataset_images += dataset_images_flicklogos47
#    dataset_query += dataset_query_flicklogos47


#---create triplets

    dataset_all_true_triplets = []
    dataset_all_false_triplets = []

    for query in dataset_query:
        for image in dataset_images:
            out_dict_triplet = {}
#            print(query)
#            print(image)
            out_dict_triplet['query'] = query
            out_dict_triplet['image'] = image

            if query['class_num'] == image['class_num']:
                dataset_all_true_triplets.append(out_dict_triplet)

            else:
                dataset_all_false_triplets.append(out_dict_triplet)

    #1.6e6   only true triplets
    print("num true triplets", len(dataset_all_true_triplets))
    print("num false triplets", len(dataset_all_false_triplets))
               
    random.seed(2)

    num_true_triplets = 180000
    num_false_triplets = 0

    dataset_true_triplets = random.sample(dataset_all_true_triplets, num_true_triplets)
    dataset_false_triplets = random.sample(dataset_all_false_triplets, num_false_triplets)

    print(len(dataset_true_triplets))
    print(len(dataset_false_triplets))

    dataset = dataset_true_triplets + dataset_false_triplets

    print("len dataset", len(dataset))

    dataset_train_val, dataset_test = train_test_split(dataset, test_size=test_size, random_state=random_state)  
    dataset_train, dataset_val = train_test_split(dataset_train_val, test_size=val_size, random_state=random_state)  
        
    #-------dataset out
    dataset_out = {}

#    dataset_out['train'] = []
#    dataset_out['val'] = []
#    dataset_out['test'] = []
#    dataset_out['labels'] = []

    dataset_out['train'] = dataset_train
    dataset_out['val'] = dataset_val
    dataset_out['test'] = dataset_test
    dataset_out['labels'] = labels_dict

    return dataset_out


if __name__ == '__main__':

    
    base_folder = '../../datasets/'
    filename_out = '../datasets/labels.json'
    
    #split validation into val and test
    val_size = 0.1    
    test_size = 0.08

    random_state = 8
    
    dataset_out = create_labels(base_folder, val_size, test_size, random_state)
    
    print("train set:", len(dataset_out['train']))
    print("val set:", len(dataset_out['val']))
    print("test set:", len(dataset_out['test']))
    
    with open(filename_out, 'w') as f:
        json.dump(dataset_out, f)

#    print("end")
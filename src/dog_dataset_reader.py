import os
import gin
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet the TensorFlow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet the TensorFlow warnings

import os
import tarfile
import random

import requests
from scipy.io import loadmat
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class DogDatasetReader:
    def __init__(self):
        self.root = "/Users/jamie/Desktop/AI/final_project/data"
        self.images_folder = os.path.join(self.root, 'Images')
        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self.save_foler = os.path.join(self.root, 'split_data')
        self.classes_list = os.listdir(self.images_folder)
        print(len(self.classes_list))
        train_class_num = 70
        val_class_num =  20
        test_class_num = 30

        random.seed(120)
        self.train_list = random.sample(self.classes_list, train_class_num)
        self.remain_list = [rem for rem in self.classes_list if rem not in self.train_list]
        self.val_list = random.sample(self.remain_list, val_class_num)
        self.test_list = [rem for rem in self.remain_list if rem not in self.val_list]

        self.train_data = self.prepare_train_data()
        self.val_data = self.prepare_val_data()
        self.test_data = self.prepare_test_data()
        

    def prepare_train_data(self):
        train_data = []
        for class_name in self.train_list:
            images = [[i, class_name] for i in os.listdir(os.path.join(self.images_folder, class_name))]
            train_data.extend(images)
            print('Train----%s' %class_name)
            
            # img_paths = [os.path.join(self.images_folder, class_name, i) for i in os.listdir(os.path.join(self.images_folder, class_name))]
            # for index, img_path in enumerate(img_paths):
            #     img = Image.open(img_path)
            #     img = img.convert('RGB')
            #     img.save(os.path.join(self.save_foler, 'images', images[index][0]), quality=100)
        print(len(train_data))
        return train_data

    def prepare_val_data(self):
        val_data = []
        for class_name in self.val_list:
            images = [[i, class_name] for i in os.listdir(os.path.join(self.images_folder, class_name))]
            val_data.extend(images)
            print('Validate----%s' %class_name)

            # img_paths = [os.path.join(self.images_folder, class_name, i) for i in os.listdir(os.path.join(self.images_folder, class_name))]
            # for index, img_path in enumerate(img_paths):
            #     img = Image.open(img_path)
            #     img = img.convert('RGB')
            #     img.save(os.path.join(self.save_foler, 'images', images[index][0]), quality=100)
        print(len(val_data))
        return val_data
            

    def prepare_test_data(self):
        test_data = []
        for class_name in self.test_list:
            if class_name == ".DS_Store":
                continue
            images = [[i, class_name] for i in os.listdir(os.path.join(self.images_folder, class_name))]
            test_data.extend(images)
            print('Test----%s' %class_name)

            # img_paths = [os.path.join(self.images_folder, class_name, i) for i in os.listdir(os.path.join(self.images_folder, class_name))]
            # for index, img_path in enumerate(img_paths):
            #     img = Image.open(img_path)
            #     img = img.convert('RGB')
            #     img.save(os.path.join(self.save_foler, 'images', images[index][0]), quality=100)
        print(len(test_data))
        return test_data
   
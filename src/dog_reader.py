import os
import os.path as path
import json
import torch
import torch.utils.data as data
import numpy as np
import random
from PIL import Image
import pdb
import csv
import sys
import torchvision.transforms as transforms
sys.dont_write_bytecode = True



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def gray_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('P')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx


class DogDataSetReader(object):
    """
       Imagefolder for miniImageNet--ravi, StanfordDog, StanfordCar and CubBird datasets.
       Images are stored in the folder of "images";
       Indexes are stored in the CSV files.
    """

    def __init__(self, data_dir="/content/CNAPs-final-project/data", mode="train", image_size=84,
                 transform=None, loader=default_loader, gray_loader=gray_loader, 
                 episode_num=1000, way_num=5, shot_num=5, query_num=5):
        
        super(DogDataSetReader, self).__init__()
        ImgTransform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform = ImgTransform
        # set the paths of the csv files
        train_csv = os.path.join(data_dir, 'train.csv')
        val_csv = os.path.join(data_dir, 'val.csv')
        test_csv = os.path.join(data_dir, 'test.csv')


        data_list = []
        e = 0
        if mode == "train":
            # store all the classes and images into a dict
            class_img_dict = {}
            with open(train_csv) as f_csv:
                f_train = csv.reader(f_csv, delimiter=',')
                for row in f_train:
                    if f_train.line_num == 1:
                        continue
                    img_name, img_class = row

                    if img_class in class_img_dict:
                        class_img_dict[img_class].append(img_name)
                    else:
                        class_img_dict[img_class]=[]
                        class_img_dict[img_class].append(img_name)
            f_csv.close()
            class_id_dict = {class_name: class_id for class_id, class_name in enumerate(class_img_dict.keys())}
            class_list = class_img_dict.keys()

            while e < episode_num:
                # construct each episode
                # construct each episode
                episode = []
                e += 1
                # temp list = 5 classes
                temp_list = random.sample(class_list, way_num)
                label_num = -1
                support_path= []
                query_path= []
                support_labels = []
                query_labels = []

                for item in temp_list:
                    #  for each class(item)
                    label_num += 1
                    id = class_id_dict[item]

                    #  imgs_set: all images in this class
                    imgs_set = class_img_dict[item]
                    # support_imgs: K shot of images from the class(image name)
                    support_imgs = random.sample(imgs_set, shot_num)
            
                    for i in range(len(support_imgs)):
                        support_labels.append(i)
                    # all other images not in support_imgs (image name)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)

                    for i in range (len(query_imgs)):
                        query_labels.append(i)
                    # the dir of support set
                    query_dir = [path.join(data_dir, '_images', i) for i in query_imgs]
                    support_dir = [path.join(data_dir, '_images', i) for i in support_imgs]
                    
                    support_path.extend(support_dir)
                    query_path.extend(query_dir)  
                    query_labels = np.array(query_labels)
                    support_labels = np.array(support_labels)
                
                data_files = {
                    "query_path": query_path,
                    "support_path": support_path,
                    "query_labels": query_labels,
                    "support_labels": support_labels
                }
                episode.append(data_files)
                data_list.append(episode)

            
        elif mode == "val":
            # store all the classes and images into a dict
            class_img_dict = {}
            with open(val_csv) as f_csv:
                f_val = csv.reader(f_csv, delimiter=',')
                for row in f_val:
                    if f_val.line_num == 1:
                        continue
                    img_name, img_class = row

                    if img_class in class_img_dict:
                        class_img_dict[img_class].append(img_name)
                    else:
                        class_img_dict[img_class]=[]
                        class_img_dict[img_class].append(img_name)
            f_csv.close()
            class_list = class_img_dict.keys()



            while e < episode_num:   # setting the episode number to 600
                # construct each episode
                episode = []
                e += 1
                # temp list = 5 classes
                temp_list = random.sample(class_list, way_num)
                label_num = -1
                support_path= []
                query_path= []
                support_labels = []
                query_labels = []

                for item in temp_list:
                    #  for each class(item)
                    label_num += 1
                    id = class_id_dict[item]

                    #  imgs_set: all images in this class
                    imgs_set = class_img_dict[item]
                    # support_imgs: K shot of images from the class(image name)
                    support_imgs = random.sample(imgs_set, shot_num)
            
                    for i in range(len(support_imgs)):
                        support_labels.append(i)
                    # all other images not in support_imgs (image name)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)

                    for i in range (len(query_imgs)):
                        query_labels.append(i)
                    # the dir of support set
                    query_dir = [path.join(data_dir, '_images', i) for i in query_imgs]
                    support_dir = [path.join(data_dir, '_images', i) for i in support_imgs]
                    
                    support_path.extend(support_dir)
                    query_path.extend(query_dir)  
                
                data_files = {
                    "query_path": query_path,
                    "support_path": support_path,
                    "query_labels": query_labels,
                    "support_labels": support_labels
                }
                episode.append(data_files)
                data_list.append(episode)
        else:

            # store all the classes and images into a dict
            class_img_dict = {}
            with open(test_csv) as f_csv:
                f_test = csv.reader(f_csv, delimiter=',')
                for row in f_test:
                    if f_test.line_num == 1:
                        continue
                    img_name, img_class = row

                    if img_class in class_img_dict:
                        class_img_dict[img_class].append(img_name)
                    else:
                        class_img_dict[img_class]=[]
                        class_img_dict[img_class].append(img_name)
            f_csv.close()
            class_list = class_img_dict.keys()


            while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                temp_list = random.sample(class_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = class_img_dict[item]
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, '_images', i) for i in query_imgs]
                    support_dir = [path.join(data_dir, '_images', i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode) 


        self.data_list = data_list
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.gray_loader = gray_loader


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query           
        '''
        image_size = self.image_size
        # len episode_files = 5
        episode_files = self.data_list[index]
        #           data_files = {
        #             "query_path": query_path,
        #             "support_path": support_path,
        #             "query_labels": query_labels,
        #             "support_labels": support_labels
        #         }
        query_images_list = []
        query_labels = episode_files['query_labels']
        support_images_list = []
        support_labels = episode_files['support_labels']
        # episode_files : 25 support images, 25 labels, 25 query images, 25 labels
        query_path = episode_files['query_path']
        support_path = episode_files['support_path']
        for i in range(len(query_path)):
            temp_query_img = self.loader(query_path[i])
                # Normalization
            if self.transform is not None:
                temp_query_img = self.transform(temp_query_img)
            query_images_list.append(temp_query_img)

            # load support images
            temp_support_img = self.loader(support_path[i])
                # Normalization
            if self.transform is not None:
                temp_support_img = self.transform(temp_support_img)
            support_images_list.append(temp_support_img)
        query_images = torch.stack(query_images_list)
        support_images = torch.stack(support_images_list)
        task_dict = {
                'context_images':query_images,
                'context_labels': query_labels, 
                'target_images': support_images, 
                'target_labels': support_labels}
        return task_dict       
        # return (query_images, query_targets, support_images, support_targets)
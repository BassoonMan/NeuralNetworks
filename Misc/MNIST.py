import numpy as np # linear algebra
import struct
from array import array
import matplotlib.pyplot as plt
from os.path  import join
from pathlib import Path

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self):
        base_path = Path(__file__).resolve().parent.parent
        input_path = join(base_path, 'Misc/MNISTImages')
        self.training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
        self.training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
        self.test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
        self.test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)    
    
    def show_images(images, title_texts):
        cols = len(images)
        rows = int(len(images)/cols) + 1
        plt.figure(figsize=(30,20))
        index = 1    
        for x in zip(images, title_texts):        
            image = x[0]        
            title_text = x[1]
            plt.subplot(rows, cols, index)        
            plt.imshow(image, cmap=plt.cm.gray)
            if (title_text != ''):
                plt.title(title_text, fontsize = 15);        
            index += 1
        plt.show()

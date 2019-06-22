import random
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from skimage import transform
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.metrics import classification_report, confusion_matrix

"""
    Preprocess image, normalizing and resizing it

    :param frame: RGBA frame
"""    
def preprocess_image(frame, image_pp_size, section = 'whole'):
    
    # Normalize Pixel Values
    normalized_frame = frame/255.0 - 0.5

    # Crop
    cropped = normalized_frame
    if section == 'header':
        x1 = 0
        x2 = normalized_frame.shape[0]
        y1 = 0
        y2 = normalized_frame.shape[1] // 3
        cropped = normalized_frame[x1:x2,y1:y2]
    elif section == 'footer':
        x1 = 0
        x2 = normalized_frame.shape[0]
        y1 = normalized_frame.shape[1] - (normalized_frame.shape[1] // 3)
        y2 = normalized_frame.shape[1]
        cropped = normalized_frame[x1:x2,y1:y2]
    elif section == 'left':
        x1 = 0
        x2 = normalized_frame.shape[0] // 2
        y1 = 0
        y2 = normalized_frame.shape[1]
        cropped = normalized_frame[x1:x2,y1:y2]
    elif section == 'right':
        x1 = normalized_frame.shape[0] // 2
        x2 = normalized_frame.shape[0]
        y1 = 0
        y2 = normalized_frame.shape[1]
        cropped = normalized_frame[x1:x2,y1:y2]
    
    # Resize
    preprocessed_frame = transform.resize(cropped, image_pp_size)
    
    # Create a 3-Channel image
    final_image = np.dstack((preprocessed_frame, preprocessed_frame, preprocessed_frame))
    
    return final_image

"""
    Create 2D label list from 1D list
    
    :param labels: 1D label list
"""

def make_labels(labels, n=16):
    np_labels = np.zeros((len(labels), n))
    
    for i in range(len(labels)):
        np_labels[i, labels[i]] = 1
    
    return np_labels

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def choices(l, k=1):
    new_list = []
    for i in range(k):
        new_list.append(random.choice(l))
    return new_list

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     print(cm)
    
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

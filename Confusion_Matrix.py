import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import torch
from PIL import Image
from os.path import join
from skimage.io import imread, imsave
from torch import nn
import numpy as np
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms

@torch.no_grad()
def get_all_preds(model, loader, device='cpu', grayscale = False):
    dtype = torch.FloatTensor
    all_preds = torch.tensor([])
    all_classes = torch.tensor([])
    model = model.to(device)
    for batch in loader:
        images, labels = batch

        if grayscale == True:
            images = np.repeat(images, 3, axis=1)  # duplicate grayscale image to 3 channels

        images = images.to(device)
        labels = labels.type(dtype)

        preds = model(images)
        preds = preds.to('cpu')

        all_classes = torch.cat(
            (all_classes, labels)
            ,dim=0
        )
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds, all_classes




def plot_confusion_matrix(model,loader, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, device='cpu', save_path='/content/cm.jpg', grayscale = False):

    all_preds, all_classes = get_all_preds(model, loader, grayscale = grayscale)
    preds = all_preds.argmax(dim=1)
    preds = preds.type(torch.FloatTensor)

    stacked = torch.stack((all_classes, preds), dim=1)
    cm = confusion_matrix(all_classes, preds)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def visualize_model(model, classes, num_images=6,dl=None, device='cpu', grayscale = False):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dl['val']):

            if grayscale == True:
                inputs = np.repeat(inputs, 3, axis=1)  # duplicate grayscale image to 3 channels

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(classes[preds[j]]))
                plt.imshow(inputs.cpu().data[j][0,:,:], cmap='gray')
                plt.show()

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
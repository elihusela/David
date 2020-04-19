import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import torch
from PIL import Image
from os.path import join
from skimage.io import imread, imsave
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms

@torch.no_grad()
def get_all_preds(model, loader, device='cpu'):
    dtype = torch.FloatTensor
    all_preds = torch.tensor([])
    all_classes = torch.tensor([])
    for batch in loader:
        images, labels = batch
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




def plot_confusion_matrix(model,loader, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, device='cpu'):

    all_preds, all_classes = get_all_preds(model, loader)
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




# plt.figure(figsize=(len(names),len(names)))
# plot_confusion_matrix(cm, names)
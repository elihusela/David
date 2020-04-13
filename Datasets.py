from PIL import Image
from os.path import join
from skimage.io import imread, imsave
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os, sys, os.path
import pandas as pd
import pickle
import urllib
import urllib.request
import skimage
import glob
import collections
import pprint
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import skimage.transform
import warnings
import tarfile

default_pathologies = ['Atelectasis',
                       'Consolidation',
                       'Infiltration',
                       'Pneumothorax',
                       'Edema',
                       'Emphysema',
                       'Fibrosis',
                       'Effusion',
                       'Pneumonia',
                       'Pleural_Thickening',
                       'Cardiomegaly',
                       'Nodule',
                       'Mass',
                       'Hernia',
                       'Lung Lesion',
                       'Fracture',
                       'Lung Opacity',
                       'Enlarged Cardiomediastinum'
                       ]

thispath = os.path.dirname(os.path.realpath(__file__))


def normalize(sample, maxval):
    """Scales images to be roughly [-1024 1024]."""
    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    # sample = sample / np.std(sample)
    return sample


def relabel_dataset(pathologies, dataset):
    """
    Reorder, remove, or add (nans) to a dataset's labels.
    Use this to align with the output of a network.
    """
    will_drop = set(dataset.pathologies).difference(pathologies)
    if will_drop != set():
        print("{} will be dropped".format(will_drop))
    new_labels = []
    dataset.pathologies = list(dataset.pathologies)
    for pathology in pathologies:
        if pathology in dataset.pathologies:
            pathology_idx = dataset.pathologies.index(pathology)
            new_labels.append(dataset.labels[:, pathology_idx])
        else:
            print("{} doesn't exist. Adding nans instead.".format(pathology))
            values = np.empty(dataset.labels.shape[0])
            values.fill(np.nan)
            new_labels.append(values)
    new_labels = np.asarray(new_labels).T

    dataset.labels = new_labels
    dataset.pathologies = pathologies


class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")


class Merge_Dataset(Dataset):
    def __init__(self, datasets, seed=0, label_concat=False):
        super(Merge_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.datasets = datasets
        self.length = 0
        self.pathologies = datasets[0].pathologies
        self.which_dataset = np.zeros(0)
        self.offset = np.zeros(0)
        currentoffset = 0
        for i, dataset in enumerate(datasets):
            self.which_dataset = np.concatenate([self.which_dataset, np.zeros(len(dataset)) + i])
            self.length += len(dataset)
            self.offset = np.concatenate([self.offset, np.zeros(len(dataset)) + currentoffset])
            currentoffset += len(dataset)
            if dataset.pathologies != self.pathologies:
                raise Exception("incorrect pathology alignment")

        if hasattr(datasets[0], 'labels'):
            self.labels = np.concatenate([d.labels for d in datasets])
        else:
            print("WARN: not adding .labels")

        self.which_dataset = self.which_dataset.astype(int)

        if label_concat:
            new_labels = np.zeros([self.labels.shape[0], self.labels.shape[1] * len(datasets)]) * np.nan
            for i, shift in enumerate(self.which_dataset):
                size = self.labels.shape[1]
                new_labels[i, shift * size:shift * size + size] = self.labels[i]
            self.labels = new_labels

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.datasets[int(self.which_dataset[idx])][idx - int(self.offset[idx])]
        item["lab"] = self.labels[idx]
        item["source"] = self.which_dataset[idx]
        return item


class FilterDataset(Dataset):
    def __init__(self, dataset, labels=None):
        super(FilterDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        #         self.idxs = np.where(np.nansum(dataset.labels, axis=1) > 0)[0]

        self.idxs = []
        if labels:
            for label in labels:
                print("filtering for ", label)

                self.idxs += list(np.where(dataset.labels[:, dataset.pathologies.index(label)] == 1)[0])
        #             singlelabel = np.nanargmax(dataset.labels[self.idxs], axis=1)
        #             subset = [k in labels for k in singlelabel]
        #             self.idxs = self.idxs[np.array(subset)]

        self.labels = self.dataset.labels[self.idxs]

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]


def NIH_downloader_by_parts(index):
    image_links = ['https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
                   'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
                   'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
                   'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
                   'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
                   'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
                   'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
                   'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
                   'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
                   'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
                   'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
                   'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz']

    images = []

    newpath = '/content/NIH_images'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    fn = '/content/NIH_images/Images_%02d.tar.gz' % (index)

    if os.path.isfile(fn):
        print("File " + fn + " already exists")
    else:
        print('downloading', fn, '...')
        urllib.request.urlretrieve(image_links[index], fn)  # download the zip file
        print("Download succesful")

    tf = tarfile.open(fn)

    tf.extractall('/content/NIH_images/')
    tf.close()

    # Remove tar file
    if os.path.exists(fn):
        os.remove(fn)
    else:
        print("Can not delete " + fn + " as it doesn't exist.")

    # append all images into a list
    # for filename in glob.glob('/content/NIH_images/images/*.png'):
    #     images.append(filename)

    return


class NIH_Dataset(Dataset):
    """
    NIH ChestX-ray8 dataset

    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(self, imgpath,
                 csvpath=os.path.join(thispath, "Data_Entry_2017.csv.gz"),
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 pure_labels=False,
                 unique_patients=True):

        super(NIH_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]

        # Remove images with view position other than PA
        self.csv = self.csv[self.csv['View Position'] == 'PA']

        # Remove multi-finding images.
        if pure_labels:
            self.csv = self.csv[~self.csv["Finding Labels"].str.contains("\|")]

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first().reset_index()

        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        # print(img_path)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)

        return {"PA": img, "lab": self.labels[idx], "idx": idx}


class COVID19_Dataset(Dataset):
    """
    COVID-19 image data collection

    Dataset: https://github.com/ieee8023/covid-chestxray-dataset

    Paper: https://arxiv.org/abs/2003.11597
    """

    def __init__(self,
                 imgpath=os.path.join(thispath, "covid-chestxray-dataset", "images"),
                 csvpath=os.path.join(thispath, "covid-chestxray-dataset", "metadata.csv"),
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 pure_labels=False,
                 unique_patients=True):

        super(COVID19_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.views = views

        # defined here to make the code easier to read
        pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus", "Pneumocystis", "Klebsiella",
                      "Chlamydophila", "Legionella"]

        self.pathologies = ["Pneumonia", "Viral Pneumonia", "Bacterial Pneumonia", "Fungal Pneumonia",
                            "No Finding"] + pneumonias
        self.pathologies = sorted(self.pathologies)

        mapping = dict()
        mapping["Pneumonia"] = pneumonias
        mapping["Viral Pneumonia"] = ["COVID-19", "SARS", "MERS"]
        mapping["Bacterial Pneumonia"] = ["Streptococcus", "Klebsiella", "Chlamydophila", "Legionella"]
        mapping["Fungal Pneumonia"] = ["Pneumocystis"]

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]

        # Keep only the frontal views.
        # idx_pa = self.csv["view"].isin(["PA", "AP", "AP Supine"])
        idx_pa = self.csv["view"].isin(self.views)
        self.csv = self.csv[idx_pa]

        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["finding"].str.contains(pathology)
            if pathology in mapping:
                for syn in mapping[pathology]:
                    # print("mapping", syn)
                    mask |= self.csv["finding"].str.contains(syn)
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={} views={}".format(len(self), self.views)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv['filename'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        # print(img_path)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)

        return {"PA": img, "lab": self.labels[idx], "idx": idx}


class ToPILImage(object):
    def __init__(self):
        self.to_pil = transforms.ToPILImage(mode="F")

    def __call__(self, x):
        return (self.to_pil(x[0]))


class XRayResizer(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return skimage.transform.resize(img, (1, self.size, self.size), mode='constant').astype(np.float32)


class XRayCenterCrop(object):

    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)


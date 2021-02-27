"""
Takes one query image file path and a list images to query,
  as well as an optional parameter for the number n of images
  to return.
Returns n most similar images.
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from tensorlayer import logging, models
from tensorflow.nn import softmax
from tensorflow.keras.backend import floatx
import tensorflow as tf
logging.set_verbosity(logging.FATAL)


##### SLOW PART HERE - RUN ASYNCHRONOUSLY WHERE POSSIBLE #####
## load VGG model
vgg = models.vgg16(pretrained=True)

## load PCA model
f = open("data/pca.pickle", "rb")
pca = pickle.load(f)
f.close()
##############################################################


def load_img(path, grayscale=False, target_size=(224, 224), show_img=False, img_type="arr", square=True):
    """ Opens image as np array. """
    img = Image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        dims = img.size
        mi, ma = sorted(dims)
        excess = ma - mi
        shift = 0
        if excess > 0:
            shift = np.random.randint(excess)
        if dims == (ma, mi):
            box = [shift, 0, mi+shift, mi]
        else:
            box = [0, shift, mi, mi+shift]
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    if show_img:
        img.show()
    if img_type == "img":
        return img
    return np.asarray(img, dtype=floatx()) / 255


def make_tree(img_paths):
    """ Embeds the images to search """
    n = len(imgs)
    imgs = [load_img(img_path) for img_path in img_paths]
    probs = np.zeros((n, 1000))
    for i, img in enumerate(imgs):
        output = vgg(img, is_train=False)
        probs[i] = softmax(output)[0].numpy()
    embeddings = pca.transform(probs)
    tree = cKDTree(embeddings)
    return tree


def make_query(q_path):
    """ Opens and embeds query image. """
    q_img = load_img(q_path)
    output = vgg(img, is_train=False)
    probs = softmax(output)[0].numpy()
    embedding = pca.transform(probs)
    return embedding


def query_from_list(q_path, img_paths, n_results):
    """ Takes a query image path and a list of image paths.
        Returns n_results closest matches.
    """ 
    imgdict = dict(enumerate(img_paths))
    query = make_query(q_path)
    tree = make_tree(img_paths)
    results = tree.query(embedding, n_results+1)[1][0]
    return [imgdict[r] for r in results[1:]]











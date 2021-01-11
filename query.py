"""
Takes one image file path passed as a command-line argument to the script,
  as well as an optional parameter for .
Returns n
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sys import argv
import pickle
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from tensorlayer import logging, models
from tensorflow.nn import softmax
from tensorflow.keras.backend import floatx
logging.set_verbosity(logging.FATAL)

image_path = argv[1]

if len(argv) > 2:
    n_results = int(argv[2])
else:
    n_results = 10

show = False
if "show" in argv:
    show = True

img_folder = "data/images/"
vgg = models.vgg16(pretrained=True)


def load_img(path, grayscale=False, target_size=(224, 224), show_img=False):
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
    return np.asarray(img, dtype=floatx()) / 255


f = open("data/imgdict.pickle", "rb")
imgdict = pickle.load(f)
f = open("data/tree.pickle", "rb")
tree = pickle.load(f)
f = open("data/pca.pickle", "rb")
pca = pickle.load(f)
f.close()

img = load_img(img_folder + image_path)

output = vgg(img, is_train=False)
probs = softmax(output)[0].numpy()
embedding = pca.transform([probs])
results = tree.query(embedding, n_results+1)[1][0]
for r in results[1:]:
    print(imgdict[r])
    if show:
        img = Image.open(img_folder + imgdict[r])
        img.show()

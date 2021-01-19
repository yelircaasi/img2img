"""
Takes one image file path passed as a command-line argument to the script,
  as well as an optional parameter for .
Returns n
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
logging.set_verbosity(logging.FATAL)


img_folder = "data/images/"
vgg = models.vgg16(pretrained=True)
n_results = 9

def load_img(path, grayscale=False, target_size=(224, 224), show_img=False, img_type="arr", square=True):
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


f = open("data/imgdict.pickle", "rb")
imgdict = pickle.load(f)
f = open("data/tree.pickle", "rb")
tree = pickle.load(f)
f = open("data/pca.pickle", "rb")
pca = pickle.load(f)
f.close()


def query_all(img_path, 
              folder=img_folder, 
              k=n_results, 
              show=False,
              output_num=False):
    
    img = load_img(folder + img_path)
    output = vgg(img, is_train=False)
    probs = softmax(output)[0].numpy()
    embedding = pca.transform([probs])
    results = tree.query(embedding, k+1)[1][0]
    matches = []
    for r in results[1:]:
        #print(imgdict[r])
        match_path = imgdict[r]
        if show:
            img = Image.open(img_folder + match_path)
            img.show()
        if output_num:
            matches.append(r)
        else:
            matches.append(match_path)
    return matches


if __name__ == "__main__":
    image_path = "."
    n_results = 1
    while image_path != "" and n_results > 0:
        image_path = input("\nPlease enter an image name or type \"r\" for a random query image, or press enter to quit: \n\t")
        if image_path.lower() == "r":
            image_path = imgdict[np.random.randint(len(imgdict))]
        elif image_path == "":
            continue
        print("Query image: ", image_path)
        img = Image.open(img_folder + image_path)
        img.show()
        n_results = int(input("How many matches would you like: \n\t"))
        matches = query_all(image_path, k=n_results, show=True)
        for n, m in enumerate(matches):
            print(f"{n}) {m}")


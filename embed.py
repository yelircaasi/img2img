from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from tensorlayer import logging, models
import tensorflow as tf
from scipy.spatial import cKDTree
from tensorflow.nn import softmax
from tensorflow.keras.backend import floatx
logging.set_verbosity(logging.FATAL)

EMB_WEIGHTS_PATH = "models/embedder.hd5"

img_folder = "data/images/"
vgg = models.vgg16(pretrained=True)


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




if __name__ == "__main__":
	assert set(imgdict.keys()) == range(len(imgdict))
	n = len(imgdict)
	rem = n % 500
	vgg_preds = np.zeros((n, 1000))
	batch = np.zeros((500, 1000))
    for k, path in imgdict.items():
    	img = load_img("data/" + path)
        output = vgg(img, is_train=False)
        probs = softmax(output)[0].numpy()
        outputs[k] = probs
        batch[k%500] = probs

    	if k % 500 == 0:
    		f = open(f"data/batch{n//500+1}.pickle", "wb")
    		pickle.dump(embedding)
    
    f = open("data/vgg_preds.pickle", "wb")
    pickle.dump(vgg_preds, f)
    
    pca = PCA(20)
    pca.fit(vgg_preds)
    embeddings = pca.transform(vgg_preds)

    f = open("data/embeddings.pickle", "wb")
    pickle.dump(embeddings, f)
    
    f = open("data/pca.pickle", "wb")
    pickle.dump(pca, f)


    tree = cKDTree(embeddings)
    f = open("data/tree.pickle", "wb")
    pickle.dump(tree, f)



import re
import pickle
from numpy import asarray, ones
from numpy.random import shuffle, randint
from itertools import permutations as perm
import tkinter as tk  
from PIL import Image, ImageTk
root = tk.Tk()  
root.title("Triplet Viewer")  
root.geometry('1250x600')

f = open("data/imgdict.pickle", "rb")
imgdict = pickle.load(f)
f.close()

img_folder = "data/images/"
#triplet_path = "data/training/triplets_pca16.txt"
triplet_path = "data/training/triplets_old7022.txt"
root.query_path = None
#root.ids = {i: None for i in range(-1, 16)}
n_images = len(imgdict)
img_size = (400, 400)


f = open(triplet_path, "r")
already = set(re.split("[,\n]", f.read().strip("\n")))
root.already = already
print("Number of pictures already used:", len(already))
"""TODO:
1) 
2) 
3) 
"""

f = open(triplet_path, "r")
triplets = [x.split(",") for x in  f.read().strip("\n").split("\n")]
f.close()
shuffle(triplets)


root.triplet = triplets[0]
root.tripnum = 0


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
            shift = randint(excess)
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
    return asarray(img, dtype=floatx()) / 255



def initialize_images():
    root.imgs = {i: ImageTk.PhotoImage(Image.fromarray(230*ones(img_size))) for i in range(3)}




def showNext():
    root.tripnum += 1
    root.triplet = triplets[root.tripnum]
    print(root.triplet)
    root.imgs = {i: ImageTk.PhotoImage(load_img(img_folder + imgdict[int(root.triplet[i])], 
                                                target_size=img_size, img_type="img")) for i in range(3)}
    update_images()



'''
def openImage():
    """
    1) Take path from file interface
    2) Open image & add to 
    3) Display image
    """
    print("$$$")
    blank_out_images()
    #TODO (1)
    #img_path = 
    img_file = filedialog.askopenfilename(initialdir = "data/images/", 
                                          title = "Select a file") 
    root.query_path = img_file
    print(f"Query image: {img_file}")
    root.imgs[-1] = ImageTk.PhotoImage(load_img(root.query_path, img_type="img"))
    imq.configure(image=root.imgs[-1])




def executeQuery():
    """
    1) 
    2) 
    3) 
    """ 
    if root.query_path is None:
        rnum = np.random.randint(n_images)
        while rnum in root.already:
            rnum = np.random.randint(n_images)
        root.rnum = rnum
        root.query_path = imgdict[rnum]
        
        print(f"Query image: {root.query_path}")
        root.imgs[-1] = ImageTk.PhotoImage(load_img(img_folder + root.query_path, img_type="img"))
        imq.configure(image=root.imgs[-1])
    matches = query_all(root.query_path, k=15, output_num=True)
    root.ids.update({-1: rnum})
    for i in range(15):
        root.ids.update({i: matches[i]})
        root.imgs[i] = ImageTk.PhotoImage(load_img(img_folder + imgdict[matches[i]], img_type="img"))
    update_images()
    root.query_path = None


def blank_out_images():
    im0.configure(image=root.imgs[-2])
    im1.configure(image=root.imgs[-2])
    im2.configure(image=root.imgs[-2])
'''


def update_images():
    #root.imgs.update({i: load_img(imgdict[root.ids[i]]) for i in range(0, 14)})
    im0.configure(image=root.imgs[0])
    im1.configure(image=root.imgs[1])
    im2.configure(image=root.imgs[2])
    



btnNext       =tk.Button( root, 
                          height=6, 
                          width=20, 
                          text="Next", 
                          command=showNext)

btnNext.grid(      row=1, column=0)

initialize_images()
im0 = tk.Label(root, image=root.imgs[ 0])
im1 = tk.Label(root, image=root.imgs[ 1])
im2 = tk.Label(root, image=root.imgs[ 2])

im0.grid(row=0, column=0)
im1.grid(row=0, column=1)
im2.grid(row=0, column=2)



root.mainloop()

if __name__ == "__main__":
    root.mainloop()
"""
    image_path = "."
    n_results = 1
    while image_path != "" and n_results > 0:
        image_path = input("Please enter an image name or type \"r\" for a random query image, or press enter to quit: \n\t")
        if image_path.lower() == "r":
            image_path = imgdict[randint(len(imgdict))]
        elif image_path == "":
            continue
        print("Query image: ", image_path)
        img = Image.open(img_folder + image_path)
        img.show()
        n_results = int(input("How many matches would you like: \n\t"))
        matches = query_all(image_path, k=n_results, show=True)
        for n, m in enumerate(matches):
            print(f"{n}) {m}")
"""

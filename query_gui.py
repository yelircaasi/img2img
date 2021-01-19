"""
GUI Layout:

 __________ __________ __________ __________
|          |          |          |          |
|          |          |          |          |
|  Query   | Result 1 | Result 2 | Result 3 |
|          |          |          |          |
|__________|__________|__________|__________| 
| Open New |          |          |          |
|__________|          |          |          |
|__________| Result 4 | Result 5 | Result 6 |
|Get Random|          |          |          |
|__________|__________|__________|__________| 
|   Sim    |          |          |          |
|__________|          |          |          |
|   Diff   | Result 7 | Result 8 | Result 9 |
|__________|          |          |          |
|___Send___|__________|__________|__________| 
"""
from query import query_all, imgdict, load_img
import numpy as np
from numpy.random import randint, choice
import tkinter as tk  
from tkinter import filedialog
from PIL import Image, ImageTk
root = tk.Tk()  
root.title("Imgage-toImage Query")  
root.geometry('1400x750')

img_folder = "data/images/"
save_path = "data/triplets_temp.txt"
root.query_path = None
n_images = len(imgdict)


def initialize_images():
    root.imgs = {i: ImageTk.PhotoImage(Image.fromarray(230*np.ones((224, 224)))) for i in range(-2, 15)}


def getTextInput():
    """
    Takes text
    """
    print("###")



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
        img_file = imgdict[np.random.randint(n_images)]
        
        print(f"Query image: {img_file}")
        root.imgs[-1] = ImageTk.PhotoImage(load_img(root.query_path, img_type="img"))
        imq.configure(image=root.imgs[-1])
    paths = query_all(root.query_path, folder="", k=15)
    for i in range(15):
        root.imgs[i] = ImageTk.PhotoImage(load_img(img_folder + paths[i], img_type="img"))
    update_images()
    root.query_path = None


def blank_out_images():
    imq.configure(image=root.imgs[-2])
    im0.configure(image=root.imgs[-2])
    im1.configure(image=root.imgs[-2])
    im2.configure(image=root.imgs[-2])
    im3.configure(image=root.imgs[-2])
    im4.configure(image=root.imgs[-2])
    im5.configure(image=root.imgs[-2])
    im6.configure(image=root.imgs[-2])
    im7.configure(image=root.imgs[-2])
    im8.configure(image=root.imgs[-2])
    im9.configure(image=root.imgs[-2])
    imA.configure(image=root.imgs[-2])
    imB.configure(image=root.imgs[-2])
    imC.configure(image=root.imgs[-2])
    imD.configure(image=root.imgs[-2])
    imE.configure(image=root.imgs[-2])


def update_images():
    #root.imgs.update({i: load_img(imgdict[root.ids[i]]) for i in range(0, 14)})
    im0.configure(image=root.imgs[ 0])
    im1.configure(image=root.imgs[ 1])
    im2.configure(image=root.imgs[ 2])
    im3.configure(image=root.imgs[ 3])
    im4.configure(image=root.imgs[ 4])
    im5.configure(image=root.imgs[ 5])
    im6.configure(image=root.imgs[ 6])
    im7.configure(image=root.imgs[ 7])
    im8.configure(image=root.imgs[ 8])
    im9.configure(image=root.imgs[ 9])
    imA.configure(image=root.imgs[10])
    imB.configure(image=root.imgs[11])
    imC.configure(image=root.imgs[12])
    imD.configure(image=root.imgs[13])
    imE.configure(image=root.imgs[14])


btnOpen       =tk.Button( root, 
                          height=6, 
                          width=20, 
                          text="Open", 
                          command=openImage)
btnQuery      =tk.Button( root, 
                          height=6, 
                          width=20, 
                          text="Search", 
                          command=executeQuery)
textInputSim  =tk.Text(   root, 
                          height=2, 
                          width=25, 
                          bg="gray95", 
                          borderwidth=5)
textInputDiff =tk.Text(   root, 
                          height=2, 
                          width=25, 
                          bg="gray95", 
                          borderwidth=5)
btnSend       =tk.Button( root, 
                          height=6, 
                          width=20, 
                          text="Send", 
                          command=getTextInput)


btnOpen.grid(      row=1, column=0)
btnQuery.grid(      row=2, column=0)
textInputSim.grid( row=3, column=0)#, columnspan=3)
textInputDiff.grid(row=4, column=0)
btnSend.grid(      row=5, column=0)

initialize_images()
imq = tk.Label(root, image=root.imgs[-1])
im0 = tk.Label(root, image=root.imgs[ 0])
im1 = tk.Label(root, image=root.imgs[ 1])
im2 = tk.Label(root, image=root.imgs[ 2])
im3 = tk.Label(root, image=root.imgs[ 3])
im4 = tk.Label(root, image=root.imgs[ 4])
im5 = tk.Label(root, image=root.imgs[ 5])
im6 = tk.Label(root, image=root.imgs[ 6])
im7 = tk.Label(root, image=root.imgs[ 7])
im8 = tk.Label(root, image=root.imgs[ 8])
im9 = tk.Label(root, image=root.imgs[ 9])
imA = tk.Label(root, image=root.imgs[10])
imB = tk.Label(root, image=root.imgs[11])
imC = tk.Label(root, image=root.imgs[12])
imD = tk.Label(root, image=root.imgs[13])
imE = tk.Label(root, image=root.imgs[14])

imq.grid(row=0, column=0)
im0.grid(row=0, column=1)
im1.grid(row=0, column=2)
im2.grid(row=0, column=3)
im3.grid(row=0, column=4)
im4.grid(row=0, column=5)
im5.grid(row=1, column=1, rowspan=2)
im6.grid(row=1, column=2, rowspan=2)
im7.grid(row=1, column=3, rowspan=2)
im8.grid(row=1, column=4, rowspan=2)
im9.grid(row=1, column=5, rowspan=2)
imA.grid(row=3, column=1, rowspan=3)
imB.grid(row=3, column=2, rowspan=3)
imC.grid(row=3, column=3, rowspan=3)
imD.grid(row=3, column=4, rowspan=3)
imE.grid(row=3, column=5, rowspan=3)


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

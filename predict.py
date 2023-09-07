# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure each string is either "ODD"
# or "EVEN" (without the quotes) depending on whether the hexadecimal number in
# the image is odd or even. Take care not to make spelling or case mistakes. Make
# sure that the length of the list returned as output is the same as the number of
# filenames that were given as input. The judge may give unexpected results if this
# convention is not followed.
import cv2
import numpy as np
import pickle
from PIL import Image
import os



def remove_lines(img):
  _, binary = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
  kernels = np.ones((5,5), np.uint8)
  dilated = cv2.dilate(binary, kernels, iterations=1)
  return dilated

def pre_processing(image):
    top_left = (0, 0)
    top_right = (image.shape[0]-1, 0)
    bottom_left = (0, image.shape[1]-1)
    bottom_right = (image.shape[0]-1, image.shape[1]-1)
    
    color_frequencies = {}
    for corner in [top_left, top_right, bottom_left, bottom_right]:
        color = image[corner]
        color_tup = tuple(color)
        if color_tup not in color_frequencies:
           color_frequencies[color_tup] = 0
        color_frequencies[color_tup] += 1

    max_color = max(color_frequencies, key=color_frequencies.get)
    R = max_color[0]
    G = max_color[1]
    B = max_color[2]

    h,w,c = image.shape
    for i in range(h):
        for j in range(w):
           if R == image[i,j,0] and G == image[i,j,1] and B == image[i,j,2]:
            image[i,j,0] = 255
            image[i,j,1] = 255
            image[i,j,2] = 255

    
    img2 = remove_lines(image)

    h,w,c = img2.shape
    for i in range(h):
        for j in range(w):
            if 255 != img2[i,j,0]:
                img2[i,j,0] = 0
                img2[i,j,1] = 0
                img2[i,j,2] = 0
            if 255 != img2[i,j,1]:
                img2[i,j,0] = 0
                img2[i,j,1] = 0
                img2[i,j,2] = 0
            if 255 != img2[i,j,2]:
                img2[i,j,0] = 0
                img2[i,j,1] = 0
                img2[i,j,2] = 0

    return img2[:,350:450,:]

def decaptcha( filenames ):
  image_size = (100,100)
  len1 = len(filenames)
  image_pred = np.zeros((len1,100,100,3))
  for i in range(len1):
    image = cv2.imread(filenames[i])
    image_pred[i] = pre_processing(image)

  img = np.array(image_pred, dtype=np.uint8) 

  for i, image in enumerate(img):
    pil_image = Image.fromarray(image)
    pil_image.save('images/image_%d.png'% i)

  images = []
  data_dir = 'images'
  for i in range(len1):
    image_path = os.path.join(data_dir,"image_" + str(i) + ".png")
    image = Image.open(image_path).convert("L")
    image = np.array(image.resize(image_size)).flatten()
    images.append(image)
    
  model_file = 'model_lr.pkl'
  with open(model_file, 'rb') as file:
    model = pickle.load(file)
  
  labels = model.predict(images)
  return labels

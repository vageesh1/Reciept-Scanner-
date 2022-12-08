#a python for summarising all the the actions that have covered in these three files 

import cv2 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import cv2 
from skimage.filters import threshold_local
import pytesseract
import re

from pytesseract import Output

def morphology(img):
  kernel = np.ones((5,5),np.uint8)
  img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 4)
  return img


def grabcut(img):
  #first we need to create a mak image to load our results 
  mask = np.zeros(img.shape[:2],np.uint8) 
  # we need to define two models named foreground and background models 
  #as our image is of 64x64 so we define in dimensions of (1,65)
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)
  #now the main part comes we need to define the rectangle for our detection 
  rect = (20,20,img.shape[1]-20,img.shape[0]-20)
  #now we can implement the grabcut algorithm 
  cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
  #this is our new mask image with four flags where 0 and 2 pixels are backfround and 1,3 are foreground it will give our main mask result 
  mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  #returning the image
  img = img*mask2[:,:,np.newaxis]
  return img
     

def countours_detection(img):
  #conversion to gray 
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #applying gaussian blurr
  gray = cv2.GaussianBlur(gray, (11, 11), 0)
  # Edge Detection using canny edge detector
  canny = cv2.Canny(gray, 0, 200)
  #dilation of image
  canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
  #making a blank canvas for countours drwaing 
  con = np.zeros_like(img) 
  #finding countors using opencv 
  contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  #sorting the countours and keeping only the largest one 
  page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
  #now drawing the countours on blank canvas 
  con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)
  return page,con


def corner_detection(page, img):
  # Blank canvas.
  con = np.zeros_like(img)
  # Loop over the contours.
  for c in page:
    # Approximate the contour.
    epsilon = 0.02 * cv2.arcLength(c, True)
    corners = cv2.approxPolyDP(c, epsilon, True)
    # If our approximated contour has four points
    if len(corners) == 4:
        break
    cv2.drawContours(con, c, -1, (0, 255, 255), 3)
    cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
  
  # Displaying the corners.
  for index, c in enumerate(corners):
    character = chr(65 + index)
    img=cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
  
  return corners,img

def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()


def dest_cordinates(pts):
  #sperating from points
  (tl, tr, br, bl) = pts
  # Finding the maximum width.
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
  # Finding the maximum height.
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
  # Final destination co-ordinates.
  destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
  return destination_corners

def aligned_img(corners,dest_cors,img):
  # Getting the homography.
  M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(dest_cors))
  # Perspective transform using homography.
  final = cv2.warpPerspective(img, M, (dest_cors[2][0], dest_cors[2][1]), flags=cv2.INTER_LINEAR)
  return final

def register_img(img):
  morphed_img=morphology(img)
  grabcut_img=grabcut(morphed_img)
  page,countours=countours_detection(grabcut_img)
  corners,corner_img=corner_detection(page,grabcut_img)
  pts=order_points(corners)
  dest_cors=dest_cordinates(pts)
  final_img=aligned_img(pts,dest_cors,img)
  return final_img


def plot_gray(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(image, cmap='Greys_r')

def plot_rgb(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255


def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255

def text_recognsiation(img):
    extracted_text = pytesseract.image_to_string(img)

def find_amounts(text):
    amounts = re.findall(r'\d+\.\d{2}\b', text)
    floats = [float(amount) for amount in amounts]
    unique = list(dict.fromkeys(floats))
    return unique

def find_amount(img):
    registered_img=register_img(img)
    gray_registered_img=bw_scanner(registered_img)
    extracted_text = pytesseract.image_to_string(img)
    amounts = find_amounts(extracted_text)
    #this will display the grand total
    print(max(amounts))


    
img=cv2.imread(r"C:\Users\hp\Downloads\recipt image.png")
find_amount(img)
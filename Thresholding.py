import cv2
import numpy as np

def relative_thresholding(img, low_percentage, hi_percentage):
    img_min = np.min(img)
    img_max = np.max(img)
    
    thresh_low = img_min + (img_max - img_min) * low_percentage
    thresh_high = img_min + (img_max - img_min) * hi_percentage
    return np.uint8((img >= thresh_low) & (img <= thresh_high)) * 255

def absolute_thresholding(img, low_absolute, hi_absolute):
    return np.uint8((img >= low_absolute) & (img <= hi_absolute)) * 255

class Thresholding:
    """ This class is for extracting relevant pixels in an image.
    """
    def __init__(self):
        """ Init Thresholding."""
        pass

    def forward(self, img):
        """ Take an image and extract all relavant pixels.
        Parameters:
            img (np.array): Input image
        Returns:
            binary (np.array): A binary image represent all positions of relavant pixels.
        """
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue = hls[:,:,0]
        lightness = hls[:,:,1]
        #saturation = hls[:,:,2]
        vue = hsv[:,:,2]

        #Right lane is white-Coloured
        right_lane = relative_thresholding(lightness, 0.8, 1.0)
        right_lane[:,:750] = 0

        #Left lane is yellow-Coloured
        left_lane = absolute_thresholding(hue, 20, 30)
        left_lane &= relative_thresholding(vue, 0.7, 1.0)
        left_lane[:,550:] = 0

        img_output = left_lane | right_lane

        return img_output
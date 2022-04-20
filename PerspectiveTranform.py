import numpy as np
import cv2


class PerspectiveTranform:
    """ a class for getting the bird eye view from normal and vice versa
    Attributes:
    src (np.array): Coordinates of 4 source points
    dst (np.array): Coordinates of 4 destination points
    M (np.array): Matrix to transform image from front view to top view
    M_inv (np.array): Matrix to transform image from top view to front view
    """
    def __init__(self):
        self.src = np.float32([(550, 460),     # top-left
                                (150, 720),     # bottom-left
                                (1200, 720),    # bottom-right
                                (770, 460)]) 
        self.dst = np.float32([(100, 0),
                                (100, 720),
                                (1100, 720),
                                (1100, 0)])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)


    def BirdView(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)


    def NormalView(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        return cv2.warpPerspective(img,self.M_inv,img_size,flags=flags)
    
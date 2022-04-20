from cv2 import perspectiveTransform
import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML
from IPython.core.display import Video
from moviepy.editor import VideoFileClip
from cameraCalibration import *
from Thresholding import *
from PerspectiveTranform import *
from draw_lanes import *

class FindLaneLines:
    def __init__(self):
        """ Init Application"""
        self.calibration = cameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = perspectiveTransform()
        self.lanelines = DrawLanes()

    def forward(self, img,debug=0):
        if (debug == 1):
            cv2.imshow("calibration", img)
            cv2.waitKey(0)

        img = self.transform.forward(img)
        first = img
        if (debug == 1):
            cv2.imshow("forward transform", img)        ##
            cv2.waitKey(0)

        img = self.thresholding.forward(img)
        if (debug == 1):
            cv2.imshow("thresholding", img)             ## colored
            cv2.waitKey(0)
            
        img = self.drawlanes.forward(img, first)
        if (debug == 1):
            cv2.imshow("drawlanes", img)
            cv2.waitKey(0)
        img = self.transform.backward(img)
        if (debug == 1):
            cv2.imshow("backward transform", img)
            cv2.waitKey(0)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        if (debug == 1):
            cv2.imshow("before_plot", out_img)
            cv2.waitKey(0)
        out_img = self.drawlanes.plot(out_img)
        if (debug == 1):
            cv2.imshow("after_plot", out_img)
            cv2.waitKey(0)
        return out_img
    
    
    def process_image(self, input_path, output_path,debug=0):
        img = mpimg.imread(input_path)
        out_img = self.forward(img,debug)
        mpimg.imsave(output_path, out_img)

    
    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

def main():

    findLaneLines = FindLaneLines()
    # for processing only image
    img_dir = ""
    img_out = ""
    debug = 0
    findLaneLines.process_image(img_dir,img_out,debug)
    # for processing a video
    vid_dir = "challenge_video.mp4"
    vid_out = "output"
    findLaneLines.process_video(vid_dir,vid_out)

if __name__ == "__main__":
    main()
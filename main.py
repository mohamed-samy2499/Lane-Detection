import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML
from IPython.core.display import Video
from moviepy.editor import VideoFileClip
from cameraCalibration import cameraCalibration
from Thresholding import *
from PerspectiveTranform import *
from draw_lanes import *
import sys
class FindLaneLines:
    def __init__(self):
        """ Init Application"""
        self.calibration = cameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTranform()
        self.lanelines = DrawLanes()

    def forward(self, img,debug=0):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        if debug == 1:
            cv2.imshow("camera calibiration",img)
            cv2.waitKey(0)
        img = self.transform.BirdView(img)
        first = img
        if debug == 1:
            cv2.imshow("Bird view",img)
            cv2.waitKey(0)
        img = self.thresholding.forward(img)
        second = img
        third = None
        if debug == 1:
            cv2.imshow("Binary image",img)
            cv2.waitKey(0)
        img,left_line,right_line,img1,topleft_windows,bottomright_windows = self.lanelines.forward(img, first, second)
        if debug == 1:
            debug_img = np.dstack((img1, img1, img1))
            cv2.polylines(debug_img,[left_line],False,(255,0,0),15)
            cv2.polylines(debug_img,[right_line],False,(255,0,0),15)
            cv2.imshow("lane lines connected",debug_img)
            cv2.waitKey(0)
            for i in range(18):
                cv2.rectangle(debug_img,topleft_windows[i],bottomright_windows[i],(0,0,255),8)
            # third = debug_img
            cv2.imshow("sliding windows",debug_img)
            cv2.waitKey(0)
            cv2.imshow("Filled lanes",img)
            cv2.waitKey(0)
        img = self.transform.NormalView(img)
        if debug == 1:
            cv2.imshow("Normal view",img)
            cv2.waitKey(0)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        if debug == 1:
            cv2.imshow("final out",out_img)
            cv2.waitKey(0)
        return out_img

    def process_image(self, input_path, output_path,debug):
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(1280,720))
        out_img = self.forward(img,debug)
        cv2.imwrite(output_path,out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

def main():

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    input_choice= int(sys.argv[3])
    debug = int(sys.argv[4])
    if input_choice ==0 :
        findLaneLines = FindLaneLines()
        findLaneLines.process_image(in_dir,out_dir,debug)
    if input_choice ==1 :
        findLaneLines = FindLaneLines()
        # findLaneLines.process_video("project_video.mp4","output_videos/output.mp4")
        findLaneLines.process_video(in_dir,out_dir)


if __name__ == "__main__":
    main()
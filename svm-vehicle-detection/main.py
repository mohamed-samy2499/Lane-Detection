
from train import processFiles, trainSVM
from detector import Detector

# from numba import jit, cuda
import cv2
# @jit(target ="cuda")   
def mainDrive():
    pos_dir = "vehicles"
    neg_dir = "non-vehicles"

    video_file = "project_video.mp4"

    feature_data_filename = "feature_data.pkl"
    # processFiles(pos_dir, neg_dir, recurse=True,output_file=True,output_filename=feature_data_filename,
    #     color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
    #     hist_features=True, spatial_features=True, hog_lib="cv",
    #     size=(64,64), pix_per_cell=(8,8), cells_per_block=(2,2),
    #     hog_bins=20, hist_bins=16, spatial_size=(20,20))
    classifier_data_filename = "classifier_data.pkl"
    # classifier_data = trainSVM(filepath=feature_data_filename,C=1000,output_file=True,output_filename=classifier_data_filename)

    detector = Detector(init_size=(90,90), x_overlap=0.7, y_step=0.01,
        x_range=(0.02, 0.98), y_range=(0.55, 0.89), scale=1.3)
    detector.loadClassifier(filepath=classifier_data_filename)

    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap, num_frames=9, threshold=120,
        draw_heatmap_size=0.3,write=True)



if __name__ == "__main__":
    mainDrive()

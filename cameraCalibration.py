import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class cameraCalibration():
    """ this class is used to calibrate camera and cancel distortion so,
    the detection is more accurate.
    We use OpenCV  documentation, functions and guidelines to do this
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    """

    def __init__(self, image_dir, nx, ny, debug=False):
        """ Init CameraCalibration.

        Parameters:
            image_dir (str): path to folder contains chessboard images
            nx (int): no of squares for chessboard width
            ny (int): no of squares for chessboard height
        """
        fnames = glob.glob("{}/*".format(image_dir))
        objpoints = []
        imgpoints = []

        # Coordinates of chessboard's corners in 3D
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Go through all chessboard images
        for f in fnames:
            img = mpimg.imread(f)

            # Convert to grayscale image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(img, (nx, ny))
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)

        shape = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, shape, None, None)

        if not ret:
            raise Exception("Unable to calibrate camera")

    def undistort(self, img):
        """ Return undistort image.

        Parameters:
            img (np.array): Input image

        Returns:
            Image (np.array): Undistorted image
        """
        # Convert to grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

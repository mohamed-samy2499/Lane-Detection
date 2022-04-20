from turtle import left
import cv2
from matplotlib.pyplot import plot
import numpy as np
import matplotlib.image as mpimg

def hist(img):
        bottom_half = img[img.shape[0]//2:,:]
        # cv2.imshow("bottom _half", bottom_half)
        # cv2.waitKey(0)
        return np.sum(bottom_half, axis=0)

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

class DrawLanes:
    """ Class includes information about detected lane lines.
    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        parameters (dict): Dictionary containing all parameters needed for the pipeline
        debug (boolean): Flag for debug/normal mode
    """
    def __init__(self):
        """Init Drawlanes.
        Parameters:
            left_fit (np.array): Coefficients of polynomial that fit left lane
            right_fit (np.array): Coefficients of polynomial that fit right lane
            binarized (np.array): binarized image
        """
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        self.left_curve_img = mpimg.imread('turn-left.png')
        self.right_curve_img = mpimg.imread('turn-right.png')
        self.keep_straight_img = mpimg.imread('straight-ahead.png')
        # self.left_curve_img = cv2.cvtColor(self.left_curve_img, cv2.CV_8U)
        # self.right_curve_img = cv2.cvtColor(self.right_curve_img, cv2.CV_8U)
        # self.keep_straight_img = cv2.cvtColor(self.keep_straight_img, cv2.CV_8U)
        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # HYPERPARAMETERS
        # Number of sliding windows
        self.nwindows = 9
        # Width of the the windows +/- margin
        self.margin = 100
        # Mininum number of pixels found to recenter window
        ## In recentered window, it requires at least 50 pixels to consider this window a part of the detected line.
        self.minpix = 50

    def forward(self, img, first, second, third):
        """Take a image and detect lane lines.
        Parameters:
            img (np.array): An binary image containing relevant pixels
        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        self.first = first
        self.second = second
        self.third = third
        self.extract_features(img)
        return self.fit_poly(img)

    
    def extract_features(self, img):
        """ Extract features from a binary image
        Parameters:
            img (np.array): A binary image
        """
        self.img = img
        # Height of windows - based on nwindows and image shape
        ## divide the image in 9 equal windows
        self.window_height = np.int(img.shape[0]//self.nwindows)

        # Identify the x and y positions of all nonzero pixel in the image
        ## to be used to draw histogram out of the binarized image
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])


    def pixels_in_window(self, center, margin, height):
        """ Return all pixel that in a specific window
        Parameters:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window
        Returns:
            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy],topleft,bottomright


    def find_lane_pixels(self, img):
            """Find lane pixels from a binary warped image.
            Parameters:
                img (np.array): A binary warped image
            Returns:
                leftx (np.array): x coordinates of left lane pixels
                lefty (np.array): y coordinates of left lane pixels
                rightx (np.array): x coordinates of right lane pixels
                righty (np.array): y coordinates of right lane pixels
                out_img (np.array): A RGB image that use to display result later on.
            """
            assert(len(img.shape) == 2)

            # Create an output image to draw on and visualize the result
            ## create 3-channel image by stacking 3 2-channel images
            ## As this out_img will be the final colored image to show         ????????
            out_img = np.dstack((img, img, img))
            # print("binarized image shape: ", img.shape)
            # print("output image shape: ", out_img.shape)

            histogram = hist(img)
            midpoint = histogram.shape[0]//2
            ## np.argmax(axis=None): returns the index of maximum value in the entire array
            leftx_base = np.argmax(histogram[:midpoint])    ## the peak's index of left lane histogram
            # print("L: ", leftx_base)
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint    ## the peak's index of right lane histogram
            # print("R: ", leftx_base)

            # Current position to be update later for each window in nwindows
            leftx_current = leftx_base
            rightx_current = rightx_base
            y_current = img.shape[0] + self.window_height//2

            # Create empty lists to reveice left and right lane pixel
            leftx, lefty, rightx, righty = [], [], [], []
            # two lists the first of topleft windows points the second of the bottomright windows points
            topleft_windows = []
            bottomright_windows = []
            # Step through the windows one by one
            for _ in range(self.nwindows):
                y_current -= self.window_height
                center_left = (leftx_current, y_current)
                center_right = (rightx_current, y_current)

                ## getting the coordinates of the pixels of left and right lane lines within the current window
                good_left_x, good_left_y,topleft,bottomright = self.pixels_in_window(center_left, self.margin, self.window_height)
                topleft_windows.append(topleft)
                bottomright_windows.append(bottomright)
                good_right_x, good_right_y,topleft,bottomright = self.pixels_in_window(center_right, self.margin, self.window_height)
                topleft_windows.append(topleft)
                bottomright_windows.append(bottomright)
                # Append these indices to the lists
                leftx.extend(good_left_x)
                lefty.extend(good_left_y)
                rightx.extend(good_right_x)
                righty.extend(good_right_y)

                if len(good_left_x) > self.minpix:
                    ## calculate the actual center of the current left window
                    leftx_current = np.int32(np.mean(good_left_x))
                if len(good_right_x) > self.minpix:
                    ## calculate the actual center of the current right window
                    rightx_current = np.int32(np.mean(good_right_x))

            ## out_img contains ONLY left lane-line and right lane-line
            # cv2.imshow("find_lane_pixels", out_img)
            # cv2.waitKey(0)
            return leftx, lefty, rightx, righty, out_img,topleft_windows,bottomright_windows


    def fit_poly(self, img):
            """Find the lane line from an image and draw it.
            Parameters:
                img (np.array): a binary warped image
            Returns:
                out_img (np.array): a RGB image that have lane line drawn on that.
            """

            leftx, lefty, rightx, righty, out_img,topleft_windows,bottomright_windows = self.find_lane_pixels(img)

            if len(lefty) > 1500:
                # print("left polyfit: ", np.polyfit(lefty, leftx, 2))
                self.left_fit = np.polyfit(lefty, leftx, 2)
            if len(righty) > 1500:
                # print("right polyfit: ", np.polyfit(righty, rightx, 2))
                self.right_fit = np.polyfit(righty, rightx, 2)

            # Generate x and y values for plotting
            maxy = img.shape[0] - 1
            miny = img.shape[0] // 3
            if len(lefty):
                ## to fill the lanes to the down left limit of the image
                maxy = max(maxy, np.max(lefty))
                miny = min(miny, np.min(lefty))

            if len(righty):
                ## to fill the lanes to the down right limit of the image
                maxy = max(maxy, np.max(righty))
                miny = min(miny, np.min(righty))

            ploty = np.linspace(miny, maxy, img.shape[0])

            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

            ploty1 = ploty.reshape((ploty.shape[0],1))
            left_fitx1 = left_fitx.reshape((left_fitx.shape[0],1))
            right_fitx1 = right_fitx.reshape((right_fitx.shape[0],1))

            left_line = np.concatenate((left_fitx1,ploty1),axis=1)
            left_line = left_line = left_line.astype('int32')
            left_line = left_line.reshape((-1,1,2))
            right_line =  np.concatenate((right_fitx1,ploty1),axis=1)
            right_line = right_line = right_line.astype('int32')
            right_line = right_line.reshape((-1,1,2))
            
            # Visualization
            for i, y in enumerate(ploty):
                l = int(left_fitx[i])
                r = int(right_fitx[i])
                y = int(y)
                cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

            lR, rR, pos = self.measure_curvature()

            ## out_img contains left lane-line and right lane-line AND the green defined region between them
            # cv2.imshow("fit_poly", out_img)
            # cv2.waitKey(0)
            return out_img,left_line,right_line,img,topleft_windows,bottomright_windows

    def plot(self, out_img):
        np.set_printoptions(precision=6, suppress=True)
        lR, rR, pos = self.measure_curvature()

        value = None
        if abs(self.left_fit[0]) > abs(self.right_fit[0]):
            value = self.left_fit[0]
        else:
            value = self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')
        
        if len(self.dir) > 10:
            self.dir.pop(0)

        W = 420
        H = 400
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        ## cyan borber
        widget[0,:] = [0, 255, 255]
        widget[-1,:] = [0, 255, 255]
        widget[:,0] = [0, 255, 255]
        widget[:,-1] = [0, 255, 255]
        out_img[:H, :W] = widget

        wtab = 425
        htab = 175
        tab = np.copy(out_img[:htab, wtab:])
        tab //=2
        ## yellow border
        tab[0,:] = [255, 255, 0]
        tab[-1,:] = [255, 255, 0]
        tab[:,0] = [255, 255, 0]
        tab[:,-1] = [255, 255, 0]
        out_img[:htab, wtab:] = tab

        x, y = 430, 0
        self.first = image_resize(self.first, width = 285, height = None, inter = cv2.INTER_AREA)
        alpha_mask = self.first[:, :, 2] / 255.0
        first_img_result = self.first[:, :, :3].copy()
        overlay_first = self.first[:, :, :3]
        overlay_image_alpha(first_img_result, overlay_first, x, y, alpha_mask)
        overlay_first = np.dstack((overlay_first, overlay_first, overlay_first))
        a, b = overlay_first[:,:,3].nonzero()
        out_img[a+5, b+215+wtab//2] = first_img_result[a, b, :3]
        # cv2.imshow(" ", out_img_result)
        # cv2.waitKey(0)

        # x, y = 430, 0
        # self.second = image_resize(self.second, width = 285, height = None, inter = cv2.INTER_AREA)
        # print(self.second.shape)
        # alpha_mask = self.second[:, :, 2] / 255.0
        # second_img_result = self.second[:, :, :3].copy()
        # overlay_second = self.second[:, :, :3]
        # overlay_image_alpha(second_img_result, overlay_second, x, y, alpha_mask)
        # print(self.second.shape)
        # self.second = np.dstack((overlay_second, overlay_second, overlay_second))
        # print(self.second.shape)
        # c, d = overlay_second[:,:,3].nonzero()
        # out_img[c+5, d+505+wtab//2] = second_img_result[a, b, :3]

        # self.second = np.dstack((self.second, self.second, self.second))
        # y, x = self.second[:,:,3].nonzero()
        # out_img[y, x+505+wtab//2] = self.second[y, x, :3]

        direction = max(set(self.dir), key = self.dir.count)
        msg = "Stay Straight"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
        if direction == 'L':
            y, x = self.left_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.left_curve_img[y, x, :3]
            msg = "Turn Left"
        if direction == 'R':
            y, x = self.right_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.right_curve_img[y, x, :3]
            msg = "Turn Right"
        if direction == 'F':
            y, x = self.keep_straight_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.keep_straight_img[y, x, :3]

        cv2.putText(out_img, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_DUPLEX , fontScale=1, color=(87, 255, 209), thickness=2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(87, 255, 209), thickness=2)

        cv2.putText(
            out_img,
            "Good Lane Keeping",
            org=(10, 330),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.2,
            color=(255, 255, 255),
            thickness=2)

        cv2.putText(
            out_img,
            "Vehicle is {:.2f} m away from center".format(pos),
            org=(10, 380),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.66,
            color=(87, 255, 209),
            thickness=2)

        return out_img

    def measure_curvature(self):
        ym = 30/720     # 0.04167
        xm = 3.7/700    # 0.00529

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature)
        left_curveR =  ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5)  / np.absolute(2*left_fit[0])
        right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        xl = np.dot(self.left_fit, [700**2, 700, 1])
        xr = np.dot(self.right_fit, [700**2, 700, 1])
        pos = (1280//2 - (xl+xr)//2)*xm
        return left_curveR, right_curveR, pos 
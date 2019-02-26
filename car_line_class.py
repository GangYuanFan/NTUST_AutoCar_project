import cv2
import numpy as np

class Lane:
    """class to receive the characteristics of each line detection"""
    def __init__(self, im):
        self.img = im
        self.danger = False

    def region_of_interest(self, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(self.img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(self.img.shape) > 2:
            channel_count = self.img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        self.img = cv2.bitwise_and(self.img, mask)
        return self.img

    def Bubble_Sorted(self, array):
        """
        Bubble sorted array 2-D only
        :param array: 2-D numpy array
        :return: alone 0-axis sorted(y depends on x)
        """
        for _ in range(array.shape[0] - 1):
            for __ in range(array.shape[0] - _ - 1):
                if array[__][0] > array[__ + 1][0]:
                    tmpX = array[__ + 1][0]
                    tmpY = array[__ + 1][1]
                    array[__ + 1][0] = array[__][0]
                    array[__ + 1][1] = array[__][1]
                    array[__][0] = tmpX
                    array[__][1] = tmpY

    def draw_lines(self, lines, midLoc, color=(0, 255, 0), thickness=4, order=2):
        try:
            # print(lines.shape)
            # reshape lines to a 2d matrix
            lines = lines.reshape(lines.shape[0], lines.shape[2])
            # create array of slopes
            slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])
            # remove junk from lists
            lines = lines[~np.isnan(lines) & ~np.isinf(lines)]
            slopes = slopes[~np.isnan(slopes) & ~np.isinf(slopes)]
            # convert lines into list of points
            lines.shape = (lines.shape[0] // 2, 2)
            # Right lane
            # move all points with negative slopes into right "lane"
            right_slopes = slopes[slopes < 0]
            right_lines = np.array(list(filter(lambda x: x[0] > midLoc, lines)))
            # Left lane
            # all positive  slopes go into left "lane"
            left_slopes = slopes[slopes > 0]
            left_lines = np.array(list(filter(lambda x: x[0] < midLoc, lines)))
            if left_lines.size != 0 and right_lines.size != 0:
                max_left_x, max_left_y = left_lines.max(axis=0)
                min_left_x, min_left_y = left_lines.min(axis=0)
                max_right_x, max_right_y = right_lines.max(axis=0)
                min_right_x, min_right_y = right_lines.min(axis=0)
                # Curve fitting approach
                # calculate polynomial fit for the points in right lane
                right_curve = np.poly1d(np.polyfit(right_lines[:, 1], right_lines[:, 0], order))
                left_curve = np.poly1d(np.polyfit(left_lines[:, 1], left_lines[:, 0], order))
                # shared ceiling on the horizon for both lines
                min_y = min(min_left_y, min_right_y)

                # use new curve function f(y) to calculate x values
                max_right_x = int(right_curve(self.img.shape[0]))
                min_right_x = int(right_curve(min_right_y))

                min_left_x = int(left_curve(self.img.shape[0]))

                r1 = (min_right_x, min_y)
                r2 = (max_right_x, self.img.shape[0])

                cv2.line(self.img, r1, r2, color, thickness)

                l1 = (max_left_x, min_y)
                l2 = (min_left_x, self.img.shape[0])

                if l1[0] - l2[0] != 0 and r1[0] - r2[0] != 0:
                    right_slope = (r1[1] - r2[1]) / (r1[0] - r2[0])
                    left_slope = (l1[1] - l2[1]) / (l1[0] - l2[0])
                    cv2.line(self.img, l1, l2, color, thickness)

                    if (left_slope < -1.0 or left_slope > -0.1) or (right_slope > 1.0 or right_slope < 0.1):
                            self.danger = 1
                    else:
                        self.danger = 0
                else:
                    self.danger = 1
        except:
            self.danger = 1
            # print('no lines')

    def hough_lines(self, rho, theta, threshold, min_line_len, max_line_gap, midLoc, order=2):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(self.img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        self.img = np.zeros((*self.img.shape, 3), dtype=np.uint8)
        self.draw_lines(lines, midLoc, order=order)
        return self.img

#header

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%matplotlib qt
#%matplotlib inline
plt.interactive(True)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # # Calculate directional gradient
    #
    # channel_0 = img[:, :, 0]
    # channel_1 = img[:, :, 1]
    # channel_2 = img[:, :, 2]
    #
    #
    # # Apply x or y gradient with the OpenCV Sobel() function
    # # and take the absolute value
    # if orient == 'x':
    #     abs_sobel_0 = np.absolute(cv2.Sobel(channel_0, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    #     abs_sobel_1 = np.absolute(cv2.Sobel(channel_1, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    #     abs_sobel_2 = np.absolute(cv2.Sobel(channel_1, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    # if orient == 'y':
    #     abs_sobel_0 = np.absolute(cv2.Sobel(channel_0, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    #     abs_sobel_1 = np.absolute(cv2.Sobel(channel_1, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    #     abs_sobel_2 = np.absolute(cv2.Sobel(channel_2, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # # Rescale back to 8 bit integer
    # scaled_sobel_0 = np.uint8(255*abs_sobel_0/np.max(abs_sobel_0))
    # scaled_sobel_1 = np.uint8(255*abs_sobel_1/np.max(abs_sobel_1))
    # scaled_sobel_2 = np.uint8(255*abs_sobel_2/np.max(abs_sobel_2))
    # # Create a copy and apply the threshold
    # grad_binary_0 = np.zeros_like(scaled_sobel_0)
    # grad_binary_1 = np.zeros_like(scaled_sobel_1)
    # grad_binary_2 = np.zeros_like(scaled_sobel_2)
    # # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    # grad_binary_0[(scaled_sobel_0 >= thresh[0]) & (scaled_sobel_0 <= thresh[1])] = 1
    # grad_binary_1[(scaled_sobel_1 >= thresh[0]) & (scaled_sobel_1 <= thresh[1])] = 1
    # grad_binary_2[(scaled_sobel_2 >= thresh[0]) & (scaled_sobel_2 <= thresh[1])] = 1
    #
    # grad_binary = np.bitwise_or(grad_binary_0,grad_binary_1,grad_binary_2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return grad_binary

def process_grad_thresholds(image):
    # Choose a Sobel kernel size
    ksize = 11 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(10, 255))

    result = np.copy(gradx)

    return result

def color_thresh(img, s_thresh=(40, 255), l_thresh=(170, 255), ld_thresh=(0, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]

    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= 20) & (h_channel <= 40)] = 1
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    s_binary = h_binary & s_binary

    s_binary2 = np.zeros_like(s_channel)
    s_binary2[(s_channel >= 0) & (s_channel <= 20)] = 1
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    l_binary = l_binary & s_binary2

    return s_binary, l_binary

def add_contour(img,color,thickness):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return

def combine_grad_color_thresh(img):
    grad_binary=process_grad_thresholds(img)
    s_binary, l_binary = color_thresh(img)

    add_contour(l_binary, 1, 7)
    grad_l_binary = grad_binary & l_binary

    add_contour(s_binary, 1, 3)
    grad_s_binary = grad_binary & s_binary

    binary_combined = np.zeros_like(grad_binary)
    binary_combined[(grad_l_binary == 1) | (grad_s_binary == 1)] = 1

    return binary_combined, grad_l_binary, grad_s_binary

def region_masking_vertices(imshape):
    #ken_factor is the factor of the field of view, determines how far the trapezoid region masking can see.
    ken_factor = 0.43
    cut_y = 60
    cut_b_trapezoid = 100
    h_mod = imshape[0]*0.95-cut_y
    h_trapezoid = h_mod*ken_factor
    a_trapezoid = (imshape[1]-2*cut_b_trapezoid)*(h_mod/2-h_trapezoid)/(h_mod/2)
    vertices_mask= np.array([[(cut_b_trapezoid,imshape[0]-cut_y),(imshape[1]/2-a_trapezoid/2, imshape[0]-h_trapezoid), (imshape[1]/2+a_trapezoid/2, imshape[0]-h_trapezoid), (imshape[1]-cut_b_trapezoid,imshape[0]-cut_y)]], dtype=np.float32)
    vertices_mask = np.rint(vertices_mask)
    return cut_y, vertices_mask

def process_perspective(image):
    # to get an image from the top view perspective
    imshape = image.shape

    cut_y, vertices_mask = region_masking_vertices(imshape)

    h_new = imshape[0]*4
    w_new = imshape[1]*1
    vertices_dst= np.array([[(0,h_new),(0,0), (w_new, 0), (w_new,h_new)]], dtype=np.float32)

    matrix_transform = cv2.getPerspectiveTransform(vertices_mask,vertices_dst)
    image_dst = cv2.warpPerspective(image,matrix_transform,(w_new,h_new),flags=cv2.INTER_LINEAR)

    matrix_transform_back = cv2.getPerspectiveTransform(vertices_dst,vertices_mask)

    return image_dst, matrix_transform_back

def find_lane_pixels_sliding_windows(histogram, binary_warped,nonzeroy,nonzerox,side):
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    if side == "left":
        x_base = leftx_base
    else:
        if side == "right":
            x_base = rightx_base
        else:
            x_base = []

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 12
    # Set the width of the windows +/- margin
    margin_base = 100
    margin = margin_base
    # Set minimum number of pixels found to recenter window
    minpix = 300

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)

    # Current positions to be updated later for each window in nwindows
    x_current = x_base

    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_mean = np.int(np.mean(nonzerox[good_inds]))
            #search the points again with the updated window position
            win_x_low = x_mean - margin
            win_x_high = x_mean + margin
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            x_current = np.int(np.mean(nonzerox[good_inds]))
            margin = margin_base
        else:
            #if search failed, increase the margin to enlarge the search field
            margin += 20

        # Append these indices to the lists
        lane_inds.append(good_inds)
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        lane_inds = np.concatenate(lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    return lane_inds

def find_lane_pixels_last_fit(nonzeroy,nonzerox):
    # Set the width of the windows +/- margin
    margin = 100

    left_fit = np.copy(Left_Lane.fit_last)
    right_fit = np.copy(Right_Lane.fit_last)
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                left_fit[1]*nonzeroy + left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                right_fit[1]*nonzeroy + right_fit[2] + margin))).nonzero()[0]

    return left_lane_inds, right_lane_inds

def polyfit(x,y):
    # Fit a second order polynomial to each using `np.polyfit`
    fit = np.polyfit(y, x, 2)
    return fit


def fit2x(fit,ploty):
    # Generate x and y values for plotting
    try:
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        fitx = 0 * ploty ** 2 + 1 * ploty
    return fitx

def find_inds(binary_warped, histogram,nonzeroy, nonzerox,):
    counter_limit = 10

    if (frame == 1) | (insanity_counter >= counter_limit):
        #search with sliding windows
        left_lane_inds = find_lane_pixels_sliding_windows(histogram, binary_warped,
                                                                                 nonzeroy, nonzerox, "left")
        right_lane_inds = find_lane_pixels_sliding_windows(histogram, binary_warped,
                                                                                    nonzeroy, nonzerox, "right")
    else:
        left_lane_inds, right_lane_inds = find_lane_pixels_last_fit(nonzeroy, nonzerox)
        # if search with last fit failed, search with sliding windows:
        if len(left_lane_inds) <= minpix_lane:
            left_lane_inds = find_lane_pixels_sliding_windows(histogram, binary_warped,
                                                                                     nonzeroy, nonzerox, "left")
        if len(right_lane_inds) <= minpix_lane:
            right_lane_inds = find_lane_pixels_sliding_windows(histogram, binary_warped,
                                                                         nonzeroy, nonzerox, "right")
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return left_lane_inds,right_lane_inds, leftx,lefty, rightx,righty

def find_fits(binary_warped,perspec_white,perspec_yellow):
    global insanity_counter
    global ndegrad
    global radius_curvature_raw

    #ploty with degradation of view distance
    ploty = np.linspace(ndegrad * degrad_factor, binary_warped.shape[0] - 1,
                               binary_warped.shape[0] - ndegrad * degrad_factor)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped[ndegrad * degrad_factor:, :].nonzero()
    nonzeroy = np.array(nonzero[0]) + ndegrad * degrad_factor
    nonzerox = np.array(nonzero[1])

    histogram = np.sum(binary_warped[binary_warped.shape[0] // 4:, :], axis=0)
    left_lane_inds, right_lane_inds,leftx,lefty,rightx,righty = find_inds(perspec_yellow,histogram,nonzeroy, nonzerox,)

    left_lane_inds_y = np.array(perspec_yellow[lefty,leftx].nonzero()[0])
    right_lane_inds_y = np.array(perspec_yellow[righty,rightx].nonzero()[0])
    left_lane_inds_w = np.array(perspec_white[lefty,leftx].nonzero()[0])
    right_lane_inds_w = np.array(perspec_white[righty,rightx].nonzero()[0])

    if ((len(left_lane_inds_y)>4*minpix_lane) | (len(left_lane_inds_y)>=len(left_lane_inds_w))):
        left_lane_inds = np.copy(left_lane_inds_y)
    else:
        left_lane_inds = np.copy(left_lane_inds_w)

    if ((len(right_lane_inds_y)>4*minpix_lane) | (len(right_lane_inds_y)>=len(right_lane_inds_w))):
        right_lane_inds = np.copy(right_lane_inds_y)
    else:
        right_lane_inds = np.copy(right_lane_inds_w)

    leftx = leftx[left_lane_inds]
    lefty = lefty[left_lane_inds]
    rightx = rightx[right_lane_inds]
    righty = righty[right_lane_inds]

    if len(left_lane_inds) == 0:
        left_fit = np.copy(Left_Lane.fit_last)
    else:
        left_fit = polyfit(leftx, lefty)
    if len(right_lane_inds) == 0:
        right_fit = np.copy(Right_Lane.fit_last)
    else:
        right_fit = polyfit(rightx, righty)

    if frame == 1:
        Left_Lane.fit_stack = np.copy(left_fit)
        Right_Lane.fit_stack = np.copy(right_fit)
        Left_Lane.fit_last = np.copy(left_fit)
        Right_Lane.fit_last = np.copy(right_fit)

    #backup fits
    left_fit_bak = np.copy(left_fit)
    right_fit_bak = np.copy(right_fit)

    left_fitx = fit2x(left_fit, ploty)
    right_fitx = fit2x(right_fit, ploty)

    left_curverad, left_radius_raw = radius_curve(left_fit, ploty)
    right_curverad, right_radius_raw = radius_curve(right_fit, ploty)

    ifsanity, ifsanity_harder = sanity_check(left_curverad, right_curverad, left_fitx, right_fitx,len(left_lane_inds),len(right_lane_inds),binary_warped,ploty,lefty,righty)

    if ifsanity:
        insanity_counter = 0
        if ifsanity_harder:
            ndegrad -= 1
            if ndegrad < 0:
                ndegrad = 0
        else:
            ndegrad += np.sign(ndegrad_neutral - ndegrad)
        Left_Lane.fit_stack = np.copy(left_fit)
        Right_Lane.fit_stack = np.copy(right_fit)
    else:
        insanity_counter += 1
        ndegrad += 1
        if ndegrad > (30):
            ndegrad = 30
        left_fit = np.copy(Left_Lane.fit_stack)
        right_fit = np.copy(Right_Lane.fit_stack)

    left_fit = pt1_filter(left_fit, Left_Lane.fit_last)
    right_fit = pt1_filter(right_fit, Right_Lane.fit_last)
    left_curverad, left_radius_raw = radius_curve(left_fit, ploty)
    right_curverad, right_radius_raw = radius_curve(right_fit, ploty)
    radius_curvature_raw = (right_radius_raw + left_radius_raw)/2

    Left_Lane.radius_of_curvature = left_curverad
    Right_Lane.radius_of_curvature = right_curverad

    Left_Lane.fit_last = np.copy(left_fit)
    Right_Lane.fit_last = np.copy(right_fit)

    left_fitx = fit2x(left_fit, ploty)
    right_fitx = fit2x(right_fit, ploty)

    # converts float arrays to integer arrays
    ploty = np.rint(ploty).astype('int32')
    left_fitx = np.rint(left_fitx).astype('int32')
    right_fitx = np.rint(right_fitx).astype('int32')

    return ploty, left_fitx, right_fitx

def radius_curve(fit,ploty):
    # Define conversions in x and y from pixels space to meters
    afactor = xscale / yscale ** 2
    bfactor = xscale / yscale
    ascaled = fit[0] * afactor
    bscaled = fit[1] * bfactor

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval_scaled = np.max(ploty) * yscale

    # Calculation of R_curve (radius of curvature)
    radius_of_curvature = ((1 + (2 * ascaled * y_eval_scaled + bscaled) ** 2) ** 1.5) / np.absolute(
        2 * ascaled)
    radius_raw = ((1 + (2 * fit[0] * np.max(ploty) + fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * fit[0])

    return radius_of_curvature, radius_raw

def unwarp(ploty,left_fitx,right_fitx,Minv,image):
    # Create an image to draw the lines on
    image_zero = image*0

    # Recast the x and y points into usable format for cv2.fillPoly()
    # pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    # pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts_left = np.stack((left_fitx, ploty),axis = 1)
    pts_right = np.stack((right_fitx, ploty),axis = 1)[-1::-1]
    pts = np.concatenate((pts_left,pts_right),axis=0).astype(float)
    pts = np.array([pts])

    pts_warp = cv2.perspectiveTransform(pts, Minv)
    cv2.fillPoly(image_zero, np.int_(pts_warp), [0,255, 0])

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, image_zero, 0.3, 0)
    return result

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.radius_of_curvature = 0
        #distance in meters of vehicle center from the line
        self.fit_stack = [np.array(False)]
        #last fit for pt1_filter
        self.fit_last = np.array([0,0,0])

def pt1_filter(u,y_last):
    t_reaction = 0.1
    t_sample = 0.1
    t_star = 1/(t_reaction/t_sample + 1)
    y_current = t_star*(u - y_last) + y_last
    return y_current

def sanity_check(left_curverad,right_curverad,left_fitx,rigth_fitx,n_leftinds,n_rightinds,binary_warped,ploty,lefty,righty):
    top_dis = rigth_fitx[0]-left_fitx[0]
    bottom_dis = rigth_fitx[-1]-left_fitx[-1]
    middle_dis = rigth_fitx[len(rigth_fitx)//2]-left_fitx[len(left_fitx)//2]
    ratio_curv = np.maximum(left_curverad,right_curverad)/np.minimum(left_curverad,right_curverad)
    dis_diff = np.absolute(bottom_dis - top_dis)
    dis_diff2 = np.absolute(2*middle_dis-top_dis-bottom_dis)


    ratio_curv_limit = 100
    dis_limit_lower = 650
    dis_limit_upper = 1000
    limit_factor = 1000000
    dis_diff_limit = limit_factor / radius_curvature_raw + 400
    dis_diff2_limit = dis_diff_limit*0.75
    result = 0
    if top_dis >= 100:
        if (n_leftinds>minpix_lane) & (n_rightinds>minpix_lane):
            if dis_diff2<=dis_diff2_limit:
                if ratio_curv<=ratio_curv_limit:
                    if (bottom_dis>=dis_limit_lower) & (bottom_dis<=dis_limit_upper):
                        if dis_diff<=dis_diff_limit:
                            result = 1

    n_leftbase = np.sum(lefty >= (binary_warped.shape[0] * 3 // 4))
    n_rightbase = np.sum(righty >= (binary_warped.shape[0] * 3 // 4))
    factor = 0.3*(1 / 4) / (1 - ploty[0] / binary_warped.shape[0])
    if n_leftbase <= (n_leftinds * factor):
        result = 0
    if n_rightbase <= (n_rightinds * factor):
        result = 0

    #harder check
    ratio_curv_limit = ratio_curv_limit / 2
    dis_limit_lower = dis_limit_lower
    dis_limit_upper = dis_limit_upper
    dis_diff_limit = dis_diff_limit / 2
    dis_diff2_limit = dis_diff2_limit / 2
    result_harder = 0

    if result:
        if dis_diff2 <= dis_diff2_limit:
            if ratio_curv <= ratio_curv_limit:
                if (bottom_dis >= dis_limit_lower) & (bottom_dis <= dis_limit_upper):
                    if dis_diff <= dis_diff_limit:
                        result_harder = 1

    return result, result_harder

def process_image(img):
    global frame
    binary_combined, grad_l_binary, grad_s_binary = combine_grad_color_thresh(img)
    perspec_bc, matrix_transform_back,  = process_perspective(binary_combined)
    perspec_white, _ = process_perspective(grad_l_binary)
    perspec_yellow, _ = process_perspective(grad_s_binary)
    ploty,left_fitx,right_fitx = find_fits(perspec_bc,perspec_white,perspec_yellow)
    unwarped = unwarp(ploty,left_fitx,right_fitx,matrix_transform_back,img)
    offset = ((right_fitx[-1] + left_fitx[-1])/2-perspec_bc.shape[1]//2)*xscale*100
    text = str(np.int16((Left_Lane.radius_of_curvature+Right_Lane.radius_of_curvature)/2)) + 'm '
    text = text + str(np.int16(offset)) + 'cm'
    cv2.putText(unwarped, text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),5)
#    plt.imsave('frame_' + video.replace('.mp4','') + '/frame_' + str(frame).zfill(3) + '.png', unwarped)
    frame += 1
    return unwarped

#global variables
yscale = 30 / 2880  # meters per pixel in y dimension
xscale = 3.7 / 800  # meters per pixel in x dimension
# Number of windows to spare, to shorten the field of view, aka for degradation.
ndegrad_neutral =10
ndegrad = ndegrad_neutral
frame = 1
insanity_counter = 0
radius_curvature_raw = 1
minpix_lane = 5000
degrad_factor = 50

# images = glob.glob('output_images/undist*.jpg')
# for idx,fname in enumerate(images):
#     img = cv2.imread(fname)
#     frame = 1
#     insanity_counter = 0
#
#     Left_Lane = Line()
#     Left_Lane.fit_stack = np.array([0,0,100])
#     Right_Lane = Line()
#     Right_Lane.fit_stack = np.array([0,0,800])
#     analyse_overall = process_image(img)
#
#     cv2.imwrite('output_images/analyse_overall'+str(idx)+'.jpg',analyse_overall)

from moviepy.editor import VideoFileClip
frame = 1
insanity_counter = 0
ratio_curv_stack = 1
dis_diff_stack = 0
dis_diff2_stack = 0
video = 'challenge_video.mp4'
output = 'out_put_' + video
#output = 'frame_' + video.replace('.mp4','') + '/frame%03d.jpg'
clip1 = VideoFileClip(video)
Left_Lane = Line()
Right_Lane = Line()
Left_Lane.fit_stack = np.array([0,0,200])
Right_Lane.fit_stack = np.array([0,0,800])
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time clip.write_videofile(output, audio=False)
clip.write_videofile(output, audio=False)
#clip.write_images_sequence(output)
#
from moviepy.editor import VideoFileClip
frame = 1
insanity_counter = 0
ratio_curv_stack = 1
dis_diff_stack = 0
dis_diff2_stack = 0
video = 'harder_challenge_video.mp4'
output = 'out_put_' + video
#output = 'frame_' + video.replace('.mp4','') + '/frame%03d.jpg'
clip1 = VideoFileClip(video)
Left_Lane = Line()
Right_Lane = Line()
Left_Lane.fit_stack = np.array([0,0,200])
Right_Lane.fit_stack = np.array([0,0,800])
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time clip.write_videofile(output, audio=False)
clip.write_videofile(output, audio=False)
#clip.write_images_sequence(output)



from moviepy.editor import VideoFileClip
video = 'project_video.mp4'
output = 'out_put_' + video
#output = 'frame_' + video.replace('.mp4','') + '/frame%03d.jpg'
clip1 = VideoFileClip(video)
Left_Lane = Line()
Right_Lane = Line()
Left_Lane.fit_stack = np.array([0,0,200])
Right_Lane.fit_stack = np.array([0,0,800])
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time clip.write_videofile(output, audio=False)
clip.write_videofile(output, audio=False)
#clip.write_images_sequence(output)
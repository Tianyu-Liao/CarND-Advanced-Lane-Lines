#header

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%matplotlib qt
#%matplotlib inline
import pickle

def show(img,cvt=1):
    img = np.uint8(img)
    if cvt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img)
    return

def prin(sth):
    print(sth)
    return

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # Convert to grayscale
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

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return dir_binary

def process_grad_thresholds(image):
    # Choose a Sobel kernel size
    ksize = 9 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 255))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=15, mag_thresh=(30, 255))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

    combined1 = np.zeros_like(dir_binary)
    combined2 = np.zeros_like(dir_binary)
    combined3 = np.zeros_like(dir_binary)
    combined1[(gradx == 1)] = 1
    combined2[((mag_binary == 1) & (dir_binary == 1))] = 1
    combined3 = cv2.bitwise_or(combined1, combined2)
    #combined = np.uint8(np.rint(combined3*255))
    result = gradx

#    show(gradx,0)
#    show(grady,0)
#   show(mag_binary,0)
#    show(dir_binary,0)
#    show(combined,0)

    image_poly = np.copy(image)
    mask_poly = np.zeros_like(gradx)
    cut_y, vertices_poly = region_masking_vertices(image.shape)
    vertices_poly = np.int32(vertices_poly)
    cv2.fillPoly(mask_poly, vertices_poly, 1)
    image_poly[mask_poly == 0,:] = [0, 0, 0]
    image_poly = cv2.addWeighted(image_poly,0.5,image,0.5,1)

    row1 = np.concatenate((image_poly,cv2.cvtColor(gradx*255,cv2.COLOR_GRAY2BGR)),axis=1)
    row2 = np.concatenate((mag_binary,dir_binary),axis=1)
    row2 = np.uint8(np.rint(row2*255))
    row2 = cv2.cvtColor(row2,cv2.COLOR_GRAY2BGR)
    row3 = np.concatenate((combined2,result),axis=1)
    row3 = np.uint8(np.rint(row3*255))
    row3 = cv2.cvtColor(row3,cv2.COLOR_GRAY2BGR)
    processed_analyse = np.concatenate((row1,row2,row3),axis=0)

#    show(processed_analyse)
    return processed_analyse, result

def color_thresh(img, s_thresh=(170, 255), l_thresh=(170, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    return s_binary, l_binary

def combine_grad_color_thresh(img):
    analyse_grad,grad_binary=process_grad_thresholds(img)
    s_binary, l_binary = color_thresh(img)

    rgb_combined = np.dstack(( np.zeros_like(grad_binary), grad_binary, s_binary)) * 255
    binary_combined = np.zeros_like(grad_binary)
    binary_combined[(grad_binary == 1) | (s_binary == 1)] = 1

    s_rgb = np.dstack((s_binary,s_binary,s_binary))*255
    l_rgb = np.dstack((l_binary,l_binary,l_binary))*255
    column = np.concatenate((s_rgb,rgb_combined),axis=1)
    analyse_grad_color = np.concatenate((analyse_grad,column),axis=0)
    return binary_combined, rgb_combined, analyse_grad_color

def region_masking_vertices(imshape):
    cut_y = 60
    cut_b_trapezoid = 100
    h_mod = imshape[0]*0.95-cut_y
    h_trapezoid = h_mod*0.43
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

def find_lane_pixels_sliding_windows(out_img,binary_warped,nonzeroy,nonzerox):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)




    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    return left_lane_inds, right_lane_inds, out_img

def find_lane_pixels_last_fit(nonzeroy,nonzerox):
    # Set the width of the windows +/- margin
    margin = 100

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    left_fit = np.copy(Left_Lane.fit_stack)
    right_fit = np.copy(Right_Lane.fit_stack)
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                right_fit[1]*nonzeroy + right_fit[2] + margin)))

    return left_lane_inds, right_lane_inds


def find_lane_pixels(binary_warped):
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    minpix_lane = 300

    if Left_Lane.fit_stack[0] == False:
        left_lane_inds, right_lane_inds, out_img = find_lane_pixels_sliding_windows(out_img, binary_warped,
                                                                                          nonzeroy, nonzerox)
    else:
        left_lane_inds, right_lane_inds = find_lane_pixels_last_fit(nonzeroy,nonzerox)
        counter_limit = 5
        if (len(left_lane_inds)<=minpix_lane | len(right_lane_inds)<=minpix_lane):
            left_lane_inds_sw, right_lane_inds_sw, out_img = find_lane_pixels_sliding_windows(out_img,binary_warped,nonzeroy,nonzerox)
            if len(left_lane_inds)<=minpix_lane:
                if Left_Lane.not_detected_counter >= counter_limit:
                    left_lane_inds = np.copy(left_lane_inds_sw)


            if len(right_lane_inds)<=minpix_lane:
                if Right_Lane.not_detected_counter >= counter_limit:
                    right_lane_inds = np.copy(right_lane_inds_sw)


#    else:
#        Left_Lane.inds_stack = np.copy(left_lane_inds)
#        Left_Lane.not_detected_counter = 0
#        Right_Lane.inds_stack = np.copy(right_lane_inds)
#        Right_Lane.not_detected_counter = 0
    if len(left_lane_inds) > minpix_lane:
        Left_Lane.inds_stack = np.copy(left_lane_inds)
        Left_Lane.not_detected_counter = 0
    else:
        left_lane_inds = Left_Lane.inds_stack
        Left_Lane.not_detected_counter += 1

    if len(right_lane_inds) > minpix_lane:
        Right_Lane.inds_stack = np.copy(right_lane_inds)
        Right_Lane.not_detected_counter = 0
    else:
        right_lane_inds = np.copy(Right_Lane.inds_stack)
        Right_Lane.not_detected_counter += 1

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    Left_Lane.fit_stack = np.copy(left_fit)
    Right_Lane.fit_stack = np.copy(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 0*ploty**2 + 1*ploty
        right_fitx = 0*ploty**2 + 1*ploty

    # Plots the left and right polynomials on the lane lines
    idx_inscope_left_fitx = (left_fitx >= 0) & (left_fitx < (binary_warped.shape[1]-0.5))
    idx_inscope_right_fitx = (right_fitx >= 0) & (right_fitx < (binary_warped.shape[1]-0.5))
    # converts float arrays to integer arrays
    ploty = np.rint(ploty).astype(int)
    left_fitx = np.rint(left_fitx).astype(int)
    right_fitx = np.rint(right_fitx).astype(int)
    #prin(np.max(right_fitx[idx_inscope_right_fitx]))
   # show(binary_warped)

    # Set the width of the windows +/- margin
    margin = 100
    window_img = np.zeros_like(out_img)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    out_img[ploty[idx_inscope_left_fitx], left_fitx[idx_inscope_left_fitx]] = [255, 255, 255]
    out_img[ploty[idx_inscope_right_fitx], right_fitx[idx_inscope_right_fitx]] = [255, 255, 255]
    result = out_img

    # Define conversions in x and y from pixels space to meters
    yscale = 30/2880 # meters per pixel in y dimension
    xscale = 3.7/800 # meters per pixel in x dimension
    afactor = xscale/yscale**2
    bfactor = xscale/yscale
    aleftscaled = left_fit[0]*afactor
    bleftscaled = left_fit[1]*bfactor
    arightscaled = right_fit[0]*afactor
    brightscaled = right_fit[1]*bfactor

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval_scaled = np.max(ploty)*yscale

    # Calculation of R_curve (radius of curvature)
    Left_Lane.radius_of_curvature = ((1 + (2*aleftscaled*y_eval_scaled + bleftscaled)**2)**1.5) / np.absolute(2*aleftscaled)
    Right_Lane.radius_of_curvature = ((1 + (2*arightscaled*y_eval_scaled + brightscaled)**2)**1.5) / np.absolute(2*arightscaled)

    return result,ploty,left_fitx,right_fitx

def unwarp(warped,ploty,left_fitx,right_fitx,Minv,image):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
#    pts = np.hstack((pts_left, pts_right))
    pts = np.concatenate((pts_left,pts_right),axis=1)

    # Draw the lane onto the warped blank image
#    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.fillPoly(warp_zero, pts, [0,255, 0])


    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(warp_zero, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #result of sanity check
        self.sanity = False
        #stack for indices
        self.inds_stack = [np.array([0,0,1])]
        #counter for not detected lane
        self.not_detected_counter = 0
        #stack for fit
        self.fit_stack = [np.array(False)]

def sanity_check(left_curverad,right_curverad,left_fitx,rigth_fitx):
    top_dis = rigth_fitx[0]-left_fitx[0]
    bottom_dis = rigth_fitx[-1]-left_fitx[-1]
    ratio_curv = np.max([left_curverad,right_curverad])/np.max([left_curverad,right_curverad])
    if numpy.absolute(left_curverad-right_curverad)<=100:
        if (rigth_fitx[1]-left_fitx[1])>=700 & (rigth_fitx[1]-left_fitx[1])<=900:
            if (rigth_fitx[-1]-left_fitx[-1])>=700 & (rigth_fitx[-1]-left_fitx[-1])<=900:
                if numpy.absolute((rigth_fitx[-1]-left_fitx[-1]) - (rigth_fitx[-1]-left_fitx[-1]))<=100:
                    result = 1;
    return result

def process_image(img):
    binary_combined, rgb_combined, analyse_grad_color = combine_grad_color_thresh(img)
    processed_analyse, matrix_transform_back = process_perspective(binary_combined)
    perspec_rgb, matrix_transform_back = process_perspective(rgb_combined)
    out_img,ploty,left_fitx,right_fitx = fit_polynomial(processed_analyse)
    unwarped = unwarp(out_img,ploty,left_fitx,right_fitx,matrix_transform_back,img)
    analyse_grad_color[:img.shape[0],img.shape[1]:2*img.shape[1],:] = np.copy(unwarped)
    analyse_overall = np.concatenate((analyse_grad_color,perspec_rgb,out_img),axis=1)
    text = str(np.int32(Left_Lane.radius_of_curvature)) + 'm  ' + str(np.int32(Right_Lane.radius_of_curvature)) + 'm'
    cv2.putText(analyse_overall, text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255),5)
    return analyse_overall

# images = glob.glob('output_images/undist*.jpg')
#
# for idx,fname in enumerate(images):
#     Left_Lane = Line()
#     Right_Lane = Line()
#     img = cv2.imread(fname)
#     analyse_overall = process_image(img)
#
#     cv2.imwrite('output_images/analyse_overall'+str(idx)+'.jpg',analyse_overall)

from moviepy.editor import VideoFileClip
video = 'challenge_video.mp4'
output = 'out_put_' + video
clip1 = VideoFileClip(video)
Left_Lane = Line()
Right_Lane = Line()
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time clip.write_videofile(output, audio=False)
clip.write_videofile(output, audio=False)
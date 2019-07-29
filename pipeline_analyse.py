#header

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%matplotlib qt
#%matplotlib inline
plt.interactive(True)

import pickle
import PIL

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # h_channel = hls[:, :, 0]
    # gray = np.copy(h_channel)
    channel_0 = img[:, :, 0]
    channel_1 = img[:, :, 1]
    channel_2 = img[:, :, 2]


    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel_0 = np.absolute(cv2.Sobel(channel_0, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_sobel_1 = np.absolute(cv2.Sobel(channel_1, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_sobel_2 = np.absolute(cv2.Sobel(channel_1, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel_0 = np.absolute(cv2.Sobel(channel_0, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        abs_sobel_1 = np.absolute(cv2.Sobel(channel_1, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        abs_sobel_2 = np.absolute(cv2.Sobel(channel_2, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel_0 = np.uint8(255*abs_sobel_0/np.max(abs_sobel_0))
    scaled_sobel_1 = np.uint8(255*abs_sobel_1/np.max(abs_sobel_1))
    scaled_sobel_2 = np.uint8(255*abs_sobel_2/np.max(abs_sobel_2))
    # Create a copy and apply the threshold
    grad_binary_0 = np.zeros_like(scaled_sobel_0)
    grad_binary_1 = np.zeros_like(scaled_sobel_1)
    grad_binary_2 = np.zeros_like(scaled_sobel_2)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary_0[(scaled_sobel_0 >= thresh[0]) & (scaled_sobel_0 <= thresh[1])] = 1
    grad_binary_1[(scaled_sobel_1 >= thresh[0]) & (scaled_sobel_1 <= thresh[1])] = 1
    grad_binary_2[(scaled_sobel_2 >= thresh[0]) & (scaled_sobel_2 <= thresh[1])] = 1

    grad_binary = np.bitwise_or(grad_binary_0,grad_binary_1,grad_binary_2)


    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # Apply x or y gradient with the OpenCV Sobel() function
    # # and take the absolute value
    # if orient == 'x':
    #     abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    # if orient == 'y':
    #     abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # # Rescale back to 8 bit integer
    # scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # # Create a copy and apply the threshold
    # grad_binary = np.zeros_like(scaled_sobel)
    # # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    # grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return grad_binary

def process_grad_thresholds(image):
    # Choose a Sobel kernel size
    ksize = 11 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(10, 255))

    result = np.copy(gradx)

    image_poly = np.copy(image)
    mask_poly = np.zeros_like(gradx)
    cut_y, vertices_poly = region_masking_vertices(image.shape)
    vertices_poly = np.int32(vertices_poly)
    cv2.fillPoly(mask_poly, vertices_poly, 1)
    image_poly[mask_poly == 0,:] = [0, 0, 0]
    image_poly = cv2.addWeighted(image_poly,0.5,image,0.5,1)

    row1 = np.concatenate((image_poly,cv2.cvtColor(gradx*255,cv2.COLOR_GRAY2BGR)),axis=1)
    row2 = np.concatenate((result,result),axis=1)
    row2 = np.uint8(np.rint(row2*255))
    row2 = cv2.cvtColor(row2,cv2.COLOR_GRAY2BGR)
    row3 = np.concatenate((gradx,result),axis=1)
    row3 = np.uint8(np.rint(row3*255))
    row3 = cv2.cvtColor(row3,cv2.COLOR_GRAY2BGR)
    processed_analyse = np.concatenate((row1,row2,row3),axis=0)

#    show(processed_analyse)
    return processed_analyse, result

def color_thresh(img, s_thresh=(40, 255), l_thresh=(170, 255), ld_thresh=(0, 100)):
    img = np.copy(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    v_channel = hsv[:,:,2]
    s_channel = hsv[:,:,1]
    h_channel = hsv[:,:,0]

    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= 20) & (h_channel <= 35)] = 1
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 40)] = 1
    s_binary = h_binary & s_binary

    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= 190) & (s_channel <= 20)] = 1

    return s_binary, v_binary

def add_contour(img,color,thickness):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return

def combine_grad_color_thresh(img):
    analyse_grad,grad_binary=process_grad_thresholds(img)
    s_binary, l_binary = color_thresh(img)

    add_contour(l_binary, 1, 1)
    grad_l_binary = grad_binary & l_binary

    add_contour(s_binary, 1, 1)
    grad_s_binary = grad_binary & s_binary

    rgb_combined = np.dstack((np.zeros_like(grad_binary), grad_s_binary, grad_l_binary)) * 255
    binary_combined = np.zeros_like(grad_binary)
    binary_combined[(grad_l_binary == 1) | (grad_s_binary == 1)] = 1

    bc_rgb = np.dstack((binary_combined,binary_combined,binary_combined))*255
    s_rgb = np.dstack((s_binary,s_binary,s_binary))*255
    l_rgb = np.dstack((l_binary,l_binary,l_binary))*255
    column = np.concatenate((s_rgb,rgb_combined),axis=1)
    analyse_grad_color = np.concatenate((analyse_grad,column),axis=0)
    analyse_grad_color[2*img.shape[0]:3*img.shape[0],:img.shape[1],:] = np.copy(l_rgb)
    analyse_grad_color[1*img.shape[0]:2*img.shape[0],:img.shape[1],:] = np.copy(bc_rgb)
    analyse_grad_color[1*img.shape[0]:2*img.shape[0],img.shape[1]:2*img.shape[1],:] = np.copy(bc_rgb)
    return binary_combined, rgb_combined, analyse_grad_color, grad_l_binary, grad_s_binary

def region_masking_vertices(imshape):
    global h_trapezoid, a_trapezoid, b_trapezoid
    #ken_factor is the factor of the field of view, determines how far the trapezoid region masking can see.
    ken_factor = 0.45
    cut_y = 60
    cut_b_trapezoid = 100
    h_mod = imshape[0]*0.95-cut_y
    h_trapezoid = h_mod*ken_factor
    a_trapezoid = (imshape[1]-2*cut_b_trapezoid)*(h_mod/2-h_trapezoid)/(h_mod/2)
    b_trapezoid = imshape[1]-2*cut_b_trapezoid
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

def find_lane_pixels_sliding_windows(histogram, out_img,binary_warped,nonzeroy,nonzerox,side):
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
    minpix = minpix_lane/20

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

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_x_low,win_y_low),
        (win_x_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_mean = np.int(np.mean(nonzerox[good_inds]))
            #search the points again with the updated window position
            win_x_low = x_mean - margin
            win_x_high = x_mean + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_x_low, win_y_low),
                          (win_x_high, win_y_high), (255, 255, 0), 2)
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

    return lane_inds, out_img

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

def polyfit(x,y,weight_polyfit):
    # Fit a second order polynomial to each using `np.polyfit`
    fit = np.polyfit(y, x, 2, w=weight_polyfit)
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

def find_inds(binary_warped,out_img, histogram,nonzeroy, nonzerox,):
    counter_limit = 10

    if (frame == 1) | (insanity_counter >= counter_limit):
        #search with sliding windows
        left_lane_inds, out_img = find_lane_pixels_sliding_windows(histogram, out_img, binary_warped,
                                                                                 nonzeroy, nonzerox, "left")
        right_lane_inds, out_img = find_lane_pixels_sliding_windows(histogram, out_img, binary_warped,
                                                                                    nonzeroy, nonzerox, "right")
    else:
        left_lane_inds, right_lane_inds = find_lane_pixels_last_fit(nonzeroy, nonzerox)
        # if search with last fit failed, search with sliding windows:
        if len(left_lane_inds) <= minpix_lane:
            left_lane_inds, out_img  = find_lane_pixels_sliding_windows(histogram, out_img, binary_warped,
                                                                                     nonzeroy, nonzerox, "left")
        if len(right_lane_inds) <= minpix_lane:
            right_lane_inds, out_img  = find_lane_pixels_sliding_windows(histogram, out_img,binary_warped,
                                                                         nonzeroy, nonzerox, "right")
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return out_img, left_lane_inds,right_lane_inds, leftx,lefty, rightx,righty

def find_fits(binary_warped,perspec_white,perspec_yellow):
    global insanity_counter
    global ndegrad
    global radius_curvature_raw
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    #ploty with degradation of view distance
    ploty = np.linspace(ndegrad * degrad_factor, binary_warped.shape[0] - 1,
                        binary_warped.shape[0] - ndegrad * degrad_factor)
    weight_polyfit_perspec = np.linspace(a_trapezoid, b_trapezoid,
                        binary_warped.shape[0])
    weight_polyfit_perspec = weight_polyfit_perspec*weight_polyfit_perspec

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped[ndegrad * degrad_factor:, :].nonzero()
    nonzeroy = np.array(nonzero[0]) + ndegrad * degrad_factor
    nonzerox = np.array(nonzero[1])

    histogram = np.sum(binary_warped[binary_warped.shape[0]*3 // 4:, :], axis=0)
    out_img, left_lane_inds, right_lane_inds,leftx,lefty,rightx,righty = find_inds(binary_warped,out_img,histogram,nonzeroy, nonzerox,)

    # left_lane_inds_y = np.array(perspec_yellow[lefty,leftx].nonzero()[0])
    # right_lane_inds_y = np.array(perspec_yellow[righty,rightx].nonzero()[0])
    # left_lane_inds_w = np.array(perspec_white[lefty,leftx].nonzero()[0])
    # right_lane_inds_w = np.array(perspec_white[righty,rightx].nonzero()[0])

    # if ((len(left_lane_inds_y)>4*minpix_lane) | (len(left_lane_inds_y)>=len(left_lane_inds_w))):
    #     left_lane_inds = np.copy(left_lane_inds_y)
    # else:
    #     left_lane_inds = np.copy(left_lane_inds_w)
    #
    # if ((len(right_lane_inds_y)>4*minpix_lane) | (len(right_lane_inds_y)>=len(right_lane_inds_w))):
    #     right_lane_inds = np.copy(right_lane_inds_y)
    # else:
    #     right_lane_inds = np.copy(right_lane_inds_w)
    #
    # leftx = leftx[left_lane_inds]
    # lefty = lefty[left_lane_inds]
    # rightx = rightx[right_lane_inds]
    # righty = righty[right_lane_inds]

##

    left_inds_mask_yellow = perspec_yellow[lefty, leftx]
    right_inds_mask_yellow = perspec_yellow[righty, rightx]
    left_inds_mask_white = perspec_white[lefty, leftx]
    right_inds_mask_white = perspec_white[righty, rightx]
    weight_inds_left = weight_polyfit_perspec[lefty]
    weight_inds_right = weight_polyfit_perspec[righty]
    weight_polyfit_left = weight_inds_left*left_inds_mask_yellow + weight_inds_left*left_inds_mask_white
    weight_polyfit_right = weight_inds_right*right_inds_mask_yellow + weight_inds_right*right_inds_mask_white
##

    if len(left_lane_inds) == 0:
        left_fit = np.copy(Left_Lane.fit_last)
    else:
#        left_fit = polyfit(leftx, lefty, weight_polyfit[lefty])
        left_fit = polyfit(leftx, lefty, weight_polyfit_left)
    if len(right_lane_inds) == 0:
        right_fit = np.copy(Right_Lane.fit_last)
    else:
#        right_fit = polyfit(rightx, righty, weight_polyfit[righty])
        right_fit = polyfit(rightx, righty, weight_polyfit_right)

    if frame == 1:
        Left_Lane.fit_stack = np.copy(left_fit)
        Right_Lane.fit_stack = np.copy(right_fit)
        Left_Lane.fit_last = np.copy(left_fit)
        Right_Lane.fit_last = np.copy(right_fit)

    #backup fits
    left_fit_bak = np.copy(left_fit)
    right_fit_bak = np.copy(right_fit)

    if (len(left_lane_inds) < minpix_lane):
        left_fit = np.copy(Left_Lane.fit_last)
    if (len(right_lane_inds) < minpix_lane):
        right_fit = np.copy(Right_Lane.fit_last)

    y_base = (binary_warped.shape[0] * 3 // 4)
    n_leftbase = np.sum(lefty >= y_base)
    n_rightbase = np.sum(righty >= y_base)

    if len(lefty)==0: lefty = np.array([0])
    if len(righty)==0: righty = np.array([0])

    left_base_percent =  (ploty[-1]-y_base)/(ploty[-1]-np.min(lefty))
    right_base_percent =  (ploty[-1]-y_base)/(ploty[-1]-np.min(righty))
    factor = 0.1
    if n_leftbase <= (len(left_lane_inds) * factor*left_base_percent):
        left_fit = np.copy(Left_Lane.fit_last)
    if n_rightbase <= (len(right_lane_inds) * factor*right_base_percent):
        right_fit = np.copy(Right_Lane.fit_last)

    left_fitx = fit2x(left_fit, ploty)
    right_fitx = fit2x(right_fit, ploty)

    left_curverad, left_radius_raw = radius_curve(left_fit, ploty)
    right_curverad, right_radius_raw = radius_curve(right_fit, ploty)

    ifsanity, sanity_text, ifsanity_harder = sanity_check(left_curverad, right_curverad, left_fitx, right_fitx)

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
        if ndegrad > (40):
            ndegrad = 40
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

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    idx_inscope_left_fitx = (left_fitx >= 0) & (left_fitx < (binary_warped.shape[1] - 0.5))
    idx_inscope_right_fitx = (right_fitx >= 0) & (right_fitx < (binary_warped.shape[1] - 0.5))
    # converts float arrays to integer arrays
    ploty = np.rint(ploty).astype('int32')
    left_fitx = np.rint(left_fitx).astype('int32')
    right_fitx = np.rint(right_fitx).astype('int32')

    # Set the width of the windows +/- margin
    margin = 100
    window_img = np.zeros_like(out_img)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    out_img[ploty[idx_inscope_left_fitx], left_fitx[idx_inscope_left_fitx]] = [255, 255, 255]
    out_img[ploty[idx_inscope_right_fitx], right_fitx[idx_inscope_right_fitx]] = [255, 255, 255]
    result = out_img

    #plot the backup lanes
    left_fitx_bak = fit2x(left_fit_bak, ploty)
    right_fitx_bak = fit2x(right_fit_bak, ploty)
    left_fitx_bak = np.rint(left_fitx_bak).astype(int)
    right_fitx_bak = np.rint(right_fitx_bak).astype(int)
    # Plots the left and right polynomials on the lane lines
    idx_inscope_left_fitx_bak = (left_fitx_bak >= 0) & (left_fitx_bak < (binary_warped.shape[1] - 0.5))
    idx_inscope_right_fitx_bak = (right_fitx_bak >= 0) & (right_fitx_bak < (binary_warped.shape[1] - 0.5))
    out_img[ploty[idx_inscope_left_fitx_bak], left_fitx_bak[idx_inscope_left_fitx_bak]] = [0, 255, 255]
    out_img[ploty[idx_inscope_right_fitx_bak], right_fitx_bak[idx_inscope_right_fitx_bak]] = [255, 255, 0]

#    sanity_text = sanity_text + "\nleft counter: " + str(Left_Lane.not_detected_counter) + "\nright counter: " + str(Right_Lane.not_detected_counter)
    sanity_text = sanity_text + "\ninsanity counter: " + str(insanity_counter) + "\nframe" + str(frame)
    sanity_text = sanity_text + "\nleft inds:"+str(len(left_lane_inds))+"\nright inds:"+str(len(right_lane_inds))
    sanity_text = sanity_text + "\nndegrad:" + str(ndegrad)
    sanity_text = sanity_text + "\nleft_r:" + str(left_radius_raw)
    sanity_text = sanity_text + "\nright_r:" + str(right_radius_raw)

    y0, dy = 100, 100
    for i, line in enumerate(sanity_text.split('\n')):
        y = y0 + i * dy
        cv2.putText(result, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),5)

    return result, ploty, left_fitx, right_fitx

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
        self.fit_stack = np.array([0,0,0])
        #last fit for pt1_filter
        self.fit_last = np.array([0,0,0])

def pt1_filter(u,y_last):
    t_reaction = 0.1
    t_sample = 0.1
    t_star = 1/(t_reaction/t_sample + 1)
    y_current = t_star*(u - y_last) + y_last
    return y_current

def sanity_check(left_curverad,right_curverad,left_fitx,right_fitx):
    top_dis = right_fitx[0]-left_fitx[0]
    bottom_dis = right_fitx[-1]-left_fitx[-1]
    middle_dis = right_fitx[len(right_fitx)//2]-left_fitx[len(left_fitx)//2]
    ratio_curv = np.maximum(left_curverad,right_curverad)/np.minimum(left_curverad,right_curverad)
    dis_diff = np.absolute(bottom_dis - top_dis)
    dis_diff2 = np.absolute(2*middle_dis-top_dis-bottom_dis)


    ratio_curv_limit = 100
    dis_limit_lower = 650
    dis_limit_upper = 1000
#    dis_diff_limit = 400
#    dis_diff2_limit = dis_diff_limit/2
    limit_factor = 1000000
    dis_diff_limit = limit_factor/radius_curvature_raw + 400
    dis_diff2_limit = dis_diff_limit*0.75
    result = 0
    if top_dis >= 100:
        if dis_diff2<=dis_diff2_limit:
            if ratio_curv<=ratio_curv_limit:
                if (bottom_dis>=dis_limit_lower) & (bottom_dis<=dis_limit_upper):
                    if dis_diff<=dis_diff_limit:
                        result = 1

    text_ddis = "\ndis_diff_limit=" + str(np.rint(dis_diff_limit))

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

    text = "sanity: " + str(result) + "sanity_h: " + str(result_harder) + "\nratio_curv=" + str(np.rint(ratio_curv)) + "\ntop_dis=" + str(np.rint(top_dis))
    text = text + "\nbottom_dis=" + str(np.rint(bottom_dis)) + "\ndis_diff=" + str(np.rint(dis_diff))
    text = text + "\ndis_diff2=" + str(np.rint(dis_diff2))
    text = text + text_ddis

    return result, text, result_harder

def process_image(img):
    global frame
    binary_combined, rgb_combined, analyse_grad_color, grad_l_binary, grad_s_binary = combine_grad_color_thresh(img)
    processed_analyse, matrix_transform_back,  = process_perspective(binary_combined)
    perspec_rgb, matrix_transform_back = process_perspective(rgb_combined)
    perspec_white, _ = process_perspective(grad_l_binary)
    perspec_yellow, _ = process_perspective(grad_s_binary)
    out_img,ploty,left_fitx,right_fitx = find_fits(processed_analyse,perspec_white,perspec_yellow)
    unwarped = unwarp(ploty,left_fitx,right_fitx,matrix_transform_back,img)
    analyse_grad_color[:img.shape[0],img.shape[1]:2*img.shape[1],:] = np.copy(unwarped)
    analyse_overall = np.concatenate((analyse_grad_color,perspec_rgb,out_img),axis=1)
    offset = ((right_fitx[-1] + left_fitx[-1])/2-out_img.shape[1]//2)*xscale*100
    text = str(np.int32(Left_Lane.radius_of_curvature)) + 'm  ' + str(np.int32(Right_Lane.radius_of_curvature)) + 'm  '
    text = text + str(np.int16(offset)) + 'cm' +  str(np.int32((Left_Lane.radius_of_curvature+Right_Lane.radius_of_curvature)/2)) + 'm '
    text = text + str(np.rint(radius_curvature_raw))
    cv2.putText(analyse_overall, text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255),5)
    output_shape = tuple(ti for ti in img.shape[1::-1])
    analyse_overall = cv2.resize(analyse_overall,output_shape)
#    plt.imsave('frame_' + video.replace('.mp4','') + '/frame_' + str(frame).zfill(3) + '.png', analyse_overall)
    frame += 1
#    plt.imshow(analyse_overall)
    return analyse_overall

import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# global variables
yscale = None  # meters per pixel in y dimension
xscale = None  # meters per pixel in x dimension
# Number of windows to spare, to shorten the field of view, aka for degradation.
ndegrad_neutral = None
ndegrad = ndegrad_neutral
frame = None
insanity_counter = None
radius_curvature_raw = None
minpix_lane = None
degrad_factor = None
h_trapezoid = None
a_trapezoid = None
b_trapezoid = None

Left_Lane = None
Right_Lane = None

def init_global():
    global yscale, xscale, ndegrad_neutral, ndegrad, frame, insanity_counter, radius_curvature_raw, minpix_lane
    global degrad_factor, h_trapezoid, a_trapezoid, b_trapezoid
    global Left_Lane, Right_Lane
    #global variables
    yscale = 30 / 2880  # meters per pixel in y dimension
    xscale = 3.7 / 800  # meters per pixel in x dimension
    # Number of windows to spare, to shorten the field of view, aka for degradation.
    ndegrad_neutral =20
    ndegrad = ndegrad_neutral
    frame = 1
    insanity_counter = 0
    radius_curvature_raw = 1
    minpix_lane = 2000
    degrad_factor = 50
    h_trapezoid = 0
    a_trapezoid = 0
    b_trapezoid = 0

    Left_Lane = Line()
    Right_Lane = Line()

video = "project_video"
#video = "harder_challenge_video"
video = "challenge_video"
images = glob.glob("origin_frame_" + video + "/frame*.jpg")
init_global()
for idx,fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    analyse_overall = process_image(img)
    analyse_overall = cv2.cvtColor(analyse_overall, cv2.COLOR_RGB2BGR)
    createFolder('./' + 'output_images_' + video + '/')
    cv2.imwrite('output_images_' + video + '/analyse_overall'+str(idx)+'.jpg',analyse_overall)
exit()

from moviepy.editor import VideoFileClip
init_global()
video = 'challenge_video.mp4'
output = 'out_put_' + video
#output = 'frame_' + video.replace('.mp4','') + '/frame%03d.jpg'
clip1 = VideoFileClip(video)
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time clip.write_videofile(output, audio=False)
clip.write_videofile(output, audio=False)
#clip.write_images_sequence(output)
#
from moviepy.editor import VideoFileClip
init_global()
video = 'harder_challenge_video.mp4'
output = 'out_put_' + video
#output = 'frame_' + video.replace('.mp4','') + '/frame%03d.jpg'
clip1 = VideoFileClip(video)
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time clip.write_videofile(output, audio=False)
clip.write_videofile(output, audio=False)
#clip.write_images_sequence(output)

from moviepy.editor import VideoFileClip
init_global()
video = 'project_video.mp4'
output = 'out_put_' + video
#output = 'frame_' + video.replace('.mp4','') + '/frame%03d.jpg'
clip1 = VideoFileClip(video)
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time clip.write_videofile(output, audio=False)
clip.write_videofile(output, audio=False)
#clip.write_images_sequence(output)


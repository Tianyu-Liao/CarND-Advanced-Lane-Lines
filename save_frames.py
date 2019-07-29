import cv2
import os
import matplotlib.pyplot as plt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


video = "harder_challenge_video"
video = "challenge_video"
frame_ind = 1
for frame_ind in range(160,180):
    cap = cv2.VideoCapture(video + ".mp4")
    total_frames = cap.get(7)
    cap.set(1,frame_ind)
    ret, frame = cap.read()
    createFolder('./' + "origin_frame_" + video + "/")
    cv2.imwrite("origin_frame_" + video + "/frame" + str(frame_ind).zfill(3)+".jpg", frame)
    #plt.imsave("origin_frame_" + video + "/frame" + str(frame_ind).zfill(3)+".jpg", frame)


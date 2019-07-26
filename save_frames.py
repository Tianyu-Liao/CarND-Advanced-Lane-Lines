import cv2
import matplotlib.pyplot as plt


video = "project_video"
frame_ind = 1
for frame_ind in range(560,600):
    cap = cv2.VideoCapture(video + ".mp4")
    total_frames = cap.get(7)
    cap.set(1, frame_ind)
    ret, frame = cap.read()
    cv2.imwrite("origin_frame_" + video + "/frame" + str(frame_ind).zfill(3)+".jpg", frame)
    #plt.imsave("origin_frame_" + video + "/frame" + str(frame_ind).zfill(3)+".jpg", frame)


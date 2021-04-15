import cv2
import os
import random
import mediapipe as mp

NUM_SEGMENTS = 21
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.3)
mpDrawing = mp.solutions.drawing_utils

def calc_center_height(landmarks):
    eyesAvg = (landmarks.landmark[1].y + landmarks.landmark[2].y + landmarks.landmark[3].y + landmarks.landmark[4].y + landmarks.landmark[5].y + landmarks.landmark[6].y) / 6.0
    feetAvg = (landmarks.landmark[27].y + landmarks.landmark[28].y + landmarks.landmark[29].y + landmarks.landmark[30].y + landmarks.landmark[31].y + landmarks.landmark[32].y) / 6.0
    xCenter = (landmarks.landmark[11].x + landmarks.landmark[12].x + landmarks.landmark[23].x + landmarks.landmark[24].x) / 4.0
    yCenter = (eyesAvg + feetAvg) / 2.0
    height = (feetAvg - eyesAvg) * 1.2

    left = landmarks.landmark[0].x
    right = landmarks.landmark[0].x
    for index in range(0,33):
        if(landmarks.landmark[index].x < left):
            left = landmarks.landmark[index].x
        if(landmarks.landmark[index].x > right):
            right = landmarks.landmark[index].x
    return height, left-0.05*height, right+0.05*height, xCenter, yCenter


video_path = ''
print('Starting...')
count = 0.0
v = [9,10,19,20]
for folder in range(6,7):
    if not os.path.exists("./"+str(folder)+"/RGB_no_background_2"):
        os.mkdir("./"+str(folder)+"/RGB_no_background_2")
    for video_index in v:
        video_name = str(folder) + '/' + str(video_index) + ".MP4"
        prefix = video_name.split('.')[0]
        label_file = prefix + ".txt"
        labelsf = open(label_file,'r')
        line_0 = labelsf.readline().strip()
        line = labelsf.readline().strip()
        labels = line_0.split(",")
        labels_steps = line.split(",")
        if(labels[0] == 'a'):
            labels[0] = '0'
        if(labels_steps[0] == 'a'):
            labels_steps[0] = '0'
        start = int(labels[0])
        end = int(labels[-1])
        duration = (int)((end - start) / NUM_SEGMENTS)
        frames = []
        for segment in range(0,NUM_SEGMENTS):
        #for segment in range(1,len(labels_steps)):
        #for segment in range(1,len(labels)):
            #equally segmented
            frame = start + segment * duration + random.randint(0,duration)
            #segmented by dance steps
            #frame = random.randint(int(labels_steps[segment-1]),int(labels_steps[segment]))
            #segmented by dance movements
            #frame = random.randint(int(labels[segment-1]),int(labels[segment]))
            frames.append(frame)
        video = cv2.VideoCapture(video_name)
        frameCount = -1
        index = 0
        while video.isOpened():
            if frameCount%2 != 0:
                for  i in range(0,7):
                    video.read()
                frameCount += 1
                continue
            success, image = video.read()
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(frame)
            frameCount += 1
            if not success:
                break
            if len(frames) < 1:
                break
            if frameCount - frames[0] < 2 and frameCount - frames[0] >= 0:
                if result.pose_landmarks == None:
                    frames[0] += 1
                    frameCount -= 1
                    continue
                height,left,right,xCenter,yCenter = calc_center_height(result.pose_landmarks)
                if height < 0:
                    frames[0] += 1
                    frameCount -= 1
                    continue
                y_start = round(yCenter*1080-(height*1080/2))
                y_end = round(yCenter*1080+(height*1080/2))
                x_start = round(left*1920)
                x_end = round(right*1920)
                if(y_start < 0):
                    y_end += 0-y_start
                    y_start = 0
                if(y_end > 1080):
                    y_start -= y_end-1080
                    y_end = 1080
                if(x_start < 0):
                    x_end += 0-x_start
                    x_start = 0
                if(x_end > 1920):
                    x_start -= x_end-1920
                    x_end = 1920
                resized = image[y_start:y_end,x_start:x_end]
                resized = cv2.resize(resized,(340,256))
                cv2.imwrite("./"+str(folder)+"/RGB_no_background_2/"+str(video_index)+"_"+str(index)+".jpg",resized)
                index += 1
                frames.remove(frames[0])
        count += 1.0
        finish = (count/140.0)*100
        print("Finished " + str(folder) + "," + str(video_index) +',' + str(finish) + '%')






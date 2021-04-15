import cv2
import mediapipe as mp
import math

from moviepy.editor import AudioFileClip
import numpy as np
#import librosa.beat
#import librosa.display
import matplotlib.pyplot as plot
print("1")


def calc_angle(px0, px1, x0, x1, py0, py1, y0, y1):
    vector1 = [x1-x0,y1-y0]
    vector2 = [px1-px0,py1-py0]
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def calc_velocity_v(px0, px1, x0, x1, py0, py1, y0, y1, t):
    angle_diff = calc_angle(px0, px1, x0, x1, py0, py1, y0, y1)
    return angle_diff / t


def calc_velocity(x0, x1, y0, y1, t):
    return math.sqrt((x1 - x0)**2 + (y1 - y0)**2) / t

def calc_acceleration(v0, v1, t):
    return (v1 - v0) / t

def calc_acceleration_changing(a0, a1, t):
    return (abs(a1) - abs(a0)) / t

def calc_center_height(landmarks):
    eyesAvg = (landmarks.landmark[1].y + landmarks.landmark[2].y + landmarks.landmark[3].y + landmarks.landmark[4].y + landmarks.landmark[5].y + landmarks.landmark[6].y) / 6.0
    feetAvg = (landmarks.landmark[27].y + landmarks.landmark[28].y + landmarks.landmark[29].y + landmarks.landmark[30].y + landmarks.landmark[31].y + landmarks.landmark[32].y) / 6.0
    xCenter = (landmarks.landmark[11].x + landmarks.landmark[12].x + landmarks.landmark[23].x + landmarks.landmark[24].x) / 4.0
    yCenter = (eyesAvg + feetAvg) / 2.0
    height = feetAvg - eyesAvg
    return height, xCenter, yCenter

def normalize_position(landmarks):
    height, xCenter, yCenter = calc_center_height(landmarks)
    index = 0
    while(index < len(landmarks.landmark)):
        landmarks.landmark[index].x += 0.5 - xCenter
        landmarks.landmark[index].y += 0.5 - yCenter
        landmarks.landmark[index].x -= 0.5
        landmarks.landmark[index].y -= 0.5
        landmarks.landmark[index].x *= 0.8 / height
        landmarks.landmark[index].y *= 0.8 / height
        landmarks.landmark[index].x += 0.5
        landmarks.landmark[index].y += 0.5
        index += 1

def calc_smoothness(height, frames, image):
    halt = 0
    smoothness = 0.0
    halt0 = 0
    smoothness0 = 0.0

    # 0lefthand, 1righthand, 2leftfoot, 3rightfoot
    xList = list()
    yList = list()
    xList0 = list()
    yList0 = list()


    
    
    
    for frame in frames:
        xList0.append((frame.landmark[15].x + frame.landmark[17].x + frame.landmark[19].x) / 3.0)
        yList0.append((frame.landmark[15].y + frame.landmark[17].y + frame.landmark[19].y) / 3.0)
        xList0.append((frame.landmark[16].x + frame.landmark[18].x + frame.landmark[20].x) / 3.0)
        yList0.append((frame.landmark[16].y + frame.landmark[18].y + frame.landmark[20].y) / 3.0)
        xList0.append((frame.landmark[27].x + frame.landmark[29].x + frame.landmark[31].x) / 3.0)
        yList0.append((frame.landmark[27].y + frame.landmark[29].y + frame.landmark[31].y) / 3.0)
        xList0.append((frame.landmark[28].x + frame.landmark[30].x + frame.landmark[32].x) / 3.0)
        yList0.append((frame.landmark[28].y + frame.landmark[30].y + frame.landmark[32].y) / 3.0)
    
    for frame in frames:
        xList.append(frame.landmark[16].x)
        yList.append(frame.landmark[16].y)
        xList.append(frame.landmark[14].x)
        yList.append(frame.landmark[14].y)
        xList.append(frame.landmark[15].x)
        yList.append(frame.landmark[15].y)
        xList.append(frame.landmark[13].x)
        yList.append(frame.landmark[13].y)
        xList.append(frame.landmark[26].x)
        yList.append(frame.landmark[26].y)
        xList.append(frame.landmark[28].x)
        yList.append(frame.landmark[28].y)
        xList.append(frame.landmark[25].x)
        yList.append(frame.landmark[25].y)
        xList.append(frame.landmark[27].x)
        yList.append(frame.landmark[27].y)



    a = ""
    c = ""
    for i in range(0,4):
        v0 = calc_velocity(xList0[0+i], xList0[4+i], yList0[0+i], yList0[4+i], defaultTime)
        v1 = calc_velocity(xList0[4+i], xList0[8+i], yList0[4+i], yList0[8+i], defaultTime)
        v2 = calc_velocity(xList0[8+i], xList0[12+i], yList0[8+i], yList0[12+i], defaultTime)
        a0 = calc_acceleration(v0, v1, defaultTime)
        a1 = calc_acceleration(v1, v2, defaultTime)
        c0 = calc_acceleration_changing(a0, a1 ,defaultTime)
        if c0 > 0:
            halt0 += 1    
        smoothness0 += abs(a1) * 1 / height

    for i in range(0,4):
        v0 = calc_velocity_v(xList[0+i*2],xList[1+i*2],xList[8+i*2],xList[9+i*2],yList[0+i*2],yList[1+i*2],yList[8+i*2],yList[9+i*2],defaultTime)
        v1 = calc_velocity_v(xList[8+i*2],xList[9+i*2],xList[16+i*2],xList[17+i*2],yList[8+i*2],yList[9+i*2],yList[16+i*2],yList[17+i*2],defaultTime)
        v2 = calc_velocity_v(xList[16+i*2],xList[17+i*2],xList[24+i*2],xList[25+i*2],yList[16+i*2],yList[17+i*2],yList[24+i*2],yList[25+i*2],defaultTime)

        #print(str(v2))
        a0 = calc_acceleration(v0, v1, defaultTime)
        a1 = calc_acceleration(v1, v2, defaultTime)
        c0 = calc_acceleration_changing(a0, a1 ,defaultTime)
        if c0 > 0:
            halt += 1    
        smoothness += abs(a1)
        a = a + str(a1) + " "
        c = c + str(c0) + " "
    #image = cv2.putText(image, a, (50,100),cv2.FONT_HERSHEY_SIMPLEX,
    #1,(0,0,255),2,cv2.LINE_AA)
    #image = cv2.putText(image, str(smoothness), (50,150),cv2.FONT_HERSHEY_SIMPLEX,
    #1,(0,0,255),2,cv2.LINE_AA)



    if  (smoothness < 20 and halt >= 1) or (smoothness0 < 5 and halt0 >= 1 ):
        return 'break'
    else:
        return 'smooth'


for folder in range(4,7):
    for file in range(11,21):
        video_name = str(folder) + '/' + str(file) + ".MP4"

        mpDrawing = mp.solutions.drawing_utils
        mpPose = mp.solutions.pose
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        prefix = video_name.split('.')[0]
        videoOut = cv2.VideoWriter(prefix + '_label' + '.mp4',fourcc,10.0,(1920,1080))

        #beatCount = 0
        #audio, freq = librosa.load('C0001bgm.wav')
        #time = np.arange(0, len(audio)) / freq
        #tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=freq, units='time')
        timeCount = 0
        #print(beat_frames)
        frameCounts = []
        frameCount = 0
        frameWindow = []
        # 15 frames per second
        defaultTime = 1.0 / 15.0

        smoothnessNum = 0
        prevSmoothness = []

        pose = mpPose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.5)
        video = cv2.VideoCapture(video_name)
        while video.isOpened():
            if frameCount%2 != 0:
                for  i in range(0,7):
                    video.read()
                frameCount += 1
                continue
            success, frame = video.read()
            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            result = pose.process(frame)
            if result.pose_landmarks == None:
                continue

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #normalize_position(result.pose_landmarks)
            mpDrawing.draw_landmarks(frame, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
            if len(frameWindow) > 3:
                height,_,_ = calc_center_height(result.pose_landmarks)
                smoothness = calc_smoothness(height, frameWindow, frame)
                if smoothness == 'break':
                    smoothnessNum = 0   
                prevCount = 0
                for prev in prevSmoothness:
                    if prev == 'break':
                        prevCount += 1
                prevSmoothness.append(smoothness)
                if prevCount >= 1:
                    smoothness = 'smooth'
                if(len(prevSmoothness) > 10):
                    prevSmoothness.pop(0)
                if smoothness == 'break':
                    frameCounts.append(frameCount)
                    print(frameCount)
                frame = cv2.putText(frame, smoothness, (800,500),cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,0,255),2,cv2.LINE_AA)
                frame = cv2.putText(frame, str(frameCount), (50,150),cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,0,255),2,cv2.LINE_AA)
                frameWindow.pop(0)
            timeCount += 1 / 15.0
            #print(timeCount)
            #if beatCount < np.size(beat_frames) and beat_frames[beatCount] <= timeCount:
                #frame = cv2.putText(frame, 'beat', (100,500),cv2.FONT_HERSHEY_SIMPLEX,
                #1,(0,0,255),2,cv2.LINE_AA)
                #beatCount += 1  
            #cv2.imshow('test', frame)
            videoOut.write(frame)
            #cv2.waitKey(25)
            frameWindow.append(result.pose_landmarks)
            frameCount += 1

            smoothnessNum += 1


        video.release()
        videoOut.release()

        prefix = video_name.split(".")
        f = open(prefix[0]+".txt",'a')
        for i in frameCounts:
            if i != frameCounts[-1]:
                f.write(str(i)+",")
            else:
                f.write(str(i))
        f.close()


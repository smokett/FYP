import cv2
import os
import random
import numpy as np

def compute_TVL1(prev,curr,bound=15): 
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    # TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32
 
    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow

NUM_SEGMENTS = 21
video_path = ''
print('Starting...')
count = 0.0
window = []
v = [9,10,19,20]
TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
for folder in range(0,7):
    if not os.path.exists("./"+str(folder)+"/OpticalFlow_steps_1"):   
        os.mkdir("./"+str(folder)+"/OpticalFlow_steps_1")
    for video_index in range(1,21):
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
            if not success:
                break
            image = cv2.resize(image,(340,256))
            window.append(image)
            if len(window) > 6:
                window.pop(0)
            frameCount += 1
            if len(frames) < 1:
                break
            if frameCount - frames[0] < 4 and frameCount - frames[0] >= 2:
                for i in range(1,6):
                    prev = window[i-1]
                    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                    curr = window[i]
                    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                    flow = compute_TVL1(prev,curr)                   
                    cv2.imwrite("./"+str(folder)+"/OpticalFlow_steps_1/"+str(video_index)+"_"+str(index)+"_"+str(i-1)+"_"+"x"+".jpg",flow[:,:,0])
                    cv2.imwrite("./"+str(folder)+"/OpticalFlow_steps_1/"+str(video_index)+"_"+str(index)+"_"+str(i-1)+"_"+"y"+".jpg",flow[:,:,1])
                index += 1
                frames.remove(frames[0])
        count += 1.0
        finish = (count/140.0) * 100.0
        print("Finished " + str(folder) + "," + str(video_index) +',' + str(finish) + '%')





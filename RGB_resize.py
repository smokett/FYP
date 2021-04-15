import cv2
import os
import random

count = 0.0
v = [9,10,19,20]

for folder in range(0,7):
    if not os.path.exists("./"+str(folder)+"/RGB_test_2_resize"):
        os.mkdir("./"+str(folder)+"/RGB_test_2_resize")
    for video_index in v:
        for segment_index in range(0,21):           
            RGB_path = str(folder) + '/' + "RGB_test_2" + '/' + str(video_index) + "_" + str(segment_index) + ".jpg"
            output_path = str(folder) + '/' + "RGB_test_2_resize" + '/' + str(video_index) + "_" + str(segment_index) + ".jpg"
            image = cv2.imread(RGB_path)
            image = cv2.resize(image,(340,256))
            image = cv2.resize(image,(224,224))
            cv2.imwrite(output_path,image)
        count += 1.0
        finish = (count/140.0) * 100.0
        print("Finished " + str(folder) + "," + str(video_index) +',  ' + str(finish) + '%')
import cv2
import os
import random

count = 0.0
v = [9,10,19,20]

for folder in range(0,7):
    if not os.path.exists("./"+str(folder)+"/OpticalFlow_test_2_resize"):   
        os.mkdir("./"+str(folder)+"/OpticalFlow_test_2_resize")
    for video_index in v:
        for segment_index in range(0,21):
            for index in range(0,5):           
                x_path = str(folder) + '/' + "OpticalFlow_test_2" + '/' + str(video_index) + "_" + str(segment_index) + '_' + str(index) + '_' + 'x' + ".jpg"
                y_path = str(folder) + '/' + "OpticalFlow_test_2" + '/' + str(video_index) + "_" + str(segment_index) + '_' + str(index) + '_' + 'y' + ".jpg"
                output_path_x = str(folder) + '/' + "OpticalFlow_test_2_resize" + '/' + str(video_index) + "_" + str(segment_index) + '_' + str(index) + '_' + 'x' + ".jpg"
                output_path_y = str(folder) + '/' + "OpticalFlow_test_2_resize" + '/' + str(video_index) + "_" + str(segment_index) + '_' + str(index) + '_' + 'y' + ".jpg"
                x = cv2.imread(x_path)
                y = cv2.imread(y_path)

                x = cv2.resize(x,(224,224))
                y = cv2.resize(y,(224,224))
                cv2.imwrite(output_path_x,x)
                cv2.imwrite(output_path_y,y)
        count += 1.0
        finish = (count/140.0) * 100.0
        print("Finished " + str(folder) + "," + str(video_index) +',  ' + str(finish) + '%')
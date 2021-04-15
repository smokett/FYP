import cv2
import os
import random

count = 0.0

for folder in range(4,7):
    if not os.path.exists("./"+str(folder)+"/OpticalFlow_steps_1_crop"):   
        os.mkdir("./"+str(folder)+"/OpticalFlow_steps_1_crop")
    for video_index in range(1,21):
        size_selection=[256,224,192,168]
        position_selection=['center','left_up','left_down','right_up','right_down']
        img_h=256
        img_w=340
        for segment_index in range(0,18):
            rand_h=size_selection[random.randint(0,len(size_selection)-1)] 
            rand_w=size_selection[random.randint(0,len(size_selection)-1)]
            position=position_selection[random.randint(0,len(position_selection)-1)]
            for index in range(0,5):           
                x_path = str(folder) + '/' + "OpticalFlow_steps_1" + '/' + str(video_index) + "_" + str(segment_index) + '_' + str(index) + '_' + 'x' + ".jpg"
                y_path = str(folder) + '/' + "OpticalFlow_steps_1" + '/' + str(video_index) + "_" + str(segment_index) + '_' + str(index) + '_' + 'y' + ".jpg"
                output_path_x = str(folder) + '/' + "OpticalFlow_steps_1_crop" + '/' + str(video_index) + "_" + str(segment_index) + '_' + str(index) + '_' + 'x' + ".jpg"
                output_path_y = str(folder) + '/' + "OpticalFlow_steps_1_crop" + '/' + str(video_index) + "_" + str(segment_index) + '_' + str(index) + '_' + 'y' + ".jpg"
                x = cv2.imread(x_path)
                y = cv2.imread(y_path)
                if position == 'center':
                    cropped_x = x[round((img_h-rand_h)/2):round((img_h+rand_h)/2),round((img_w-rand_w)/2):round((img_w+rand_w)/2)]
                    cropped_y = y[round((img_h-rand_h)/2):round((img_h+rand_h)/2),round((img_w-rand_w)/2):round((img_w+rand_w)/2)]
                elif position == 'left_up':
                    cropped_x = x[0:rand_h,0:rand_w]
                    cropped_y = y[0:rand_h,0:rand_w]
                elif position == 'left_down':
                    cropped_x = x[img_h-rand_h:img_h,0:rand_w]
                    cropped_y = y[img_h-rand_h:img_h,0:rand_w]
                elif position == 'right_up':
                    cropped_x = x[0:rand_h,img_w-rand_w:img_w]
                    cropped_y = y[0:rand_h,img_w-rand_w:img_w]
                elif position == 'right_down':
                    cropped_x = x[img_h-rand_h:img_h,img_w-rand_w:img_w]
                    cropped_y = y[img_h-rand_h:img_h,img_w-rand_w:img_w]

                cropped_x = cv2.resize(cropped_x,(224,224))
                cropped_y = cv2.resize(cropped_y,(224,224))
                cv2.imwrite(output_path_x,cropped_x)
                cv2.imwrite(output_path_y,cropped_y)
        count += 1.0
        finish = (count/140.0) * 100.0
        print("Finished " + str(folder) + "," + str(video_index) +',  ' + str(finish) + '%')
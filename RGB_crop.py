import cv2
import os
import random

count = 0.0

for folder in range(0,7):
    if not os.path.exists("./"+str(folder)+"/RGB_no_background_movements_crop"):
        os.mkdir("./"+str(folder)+"/RGB_no_background_movements_crop")
    for video_index in range(1,21):
        size_selection=[256,224,192,168]
        position_selection=['center','left_up','left_down','right_up','right_down']
        img_h=256
        img_w=340
        for segment_index in range(0,31):           
            RGB_path = str(folder) + '/' + "RGB_no_background_movements" + '/' + str(video_index) + "_" + str(segment_index) + ".jpg"
            output_path = str(folder) + '/' + "RGB_no_background_movements_crop" + '/' + str(video_index) + "_" + str(segment_index) + ".jpg"
            image = cv2.imread(RGB_path)
            image = cv2.resize(image,(340,256))
            rand_h=size_selection[random.randint(0,len(size_selection)-1)] 
            rand_w=size_selection[random.randint(0,len(size_selection)-1)]
            position=position_selection[random.randint(0,len(position_selection)-1)]
            if position == 'center':
                cropped = image[round((img_h-rand_h)/2):round((img_h+rand_h)/2),round((img_w-rand_w)/2):round((img_w+rand_w)/2)]
            elif position == 'left_up':
                cropped = image[0:rand_h,0:rand_w]
            elif position == 'left_down':
                cropped = image[img_h-rand_h:img_h,0:rand_w]
            elif position == 'right_up':
                cropped = image[0:rand_h,img_w-rand_w:img_w]
            elif position == 'right_down':
                cropped = image[img_h-rand_h:img_h,img_w-rand_w:img_w]
            cropped = cv2.resize(cropped,(224,224))
            cv2.imwrite(output_path,cropped)
        count += 1.0
        finish = (count/140.0) * 100.0
        print("Finished " + str(folder) + "," + str(video_index) +',  ' + str(finish) + '%')


            


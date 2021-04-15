import os

for video in range(1,21):
    for segment in range(0,21):
        for index in range(0,5):
            path_1 = "./"+"4/"+"OpticalFlow/"+str(video)+'_'+str(segment)+'_'+str(index)+'_x.jpg'
            path_2 = "./"+"4/"+"OpticalFlow/"+str(video)+'_'+str(segment)+'_'+str(index)+'_y.jpg'
            if not os.path.exists(path_1):
                print(path_1)
            if not os.path.exists(path_2):
                print(path_2)
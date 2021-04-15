year = [17,5,3,4,1,1,5]
score = open("score.txt","r")
f = open("diving_dataset.txt",'w')
scores = []
for line in score:
    scores.append(float(line))
list = []
#diving
# for video_0 in range(0,296):
#     for video_1 in range(0,296):
#         if scores[video_0] > scores[video_1]:
#             label = 1.0
#         elif scores[video_0] < scores[video_1]:
#             label = -1.0
#         elif scores[video_0] == scores[video_1]:
#             label = 0.0
#         list.append(str(video_0+1)+","+str(video_1+1)+","+str(label))

#dance
for dancer_0 in range(0,7):
    for dancer_1 in range(0,7):
        for video_0 in range(1,21):
            for video_1 in range(0,21):
                if year[dancer_0] > year[dancer_1]:
                    label = 1.0
                elif year[dancer_0] < year[dancer_1]:
                    label = -1.0
                elif year[dancer_0] == year[dancer_1]:
                    label = 0.0
                list.append(str(dancer_0)+","+str(video_0)+","+str(dancer_1)+","+str(video_1)+","+str(label)) 
for line in list:
    f.write(line+"\n")
f.close()
import math

rgb = open("test1_rgb.txt","r")
rgb_list = []
for line in rgb:
    rgb_list.append(float(line.strip()))

optical = open("test1_opticalflow.txt","r")
optical_list = []
for line in optical:
    optical_list.append(float(line.strip()))

year = [17,5,3,4,1,1,5]
rgb_count = 0
rgb_correct = 0
for v in range(0,28):
    for k in range(v,28):
        s1 = rgb_list[math.floor(v/4)]
        s2 = rgb_list[math.floor(k/4)]
        if(year[math.floor(v/4)] != year[math.floor(k/4)]):
            rgb_count += 1
            if((year[math.floor(v/4)] > year[math.floor(k/4)] and s1 > s2)
            or (year[math.floor(v/4)] < year[math.floor(k/4)] and s1 < s2)):
                rgb_correct += 1
print("spatial only: "+str(rgb_correct/rgb_count))

optical_count = 0
optical_correct = 0
for v in range(0,28):
    for k in range(v,28):
        s1 = optical_list[math.floor(v/4)]
        s2 = optical_list[math.floor(k/4)]
        if(year[math.floor(v/4)] != year[math.floor(k/4)]):
            optical_count += 1
            if((year[math.floor(v/4)] > year[math.floor(k/4)] and s1 > s2)
            or (year[math.floor(v/4)] < year[math.floor(k/4)] and s1 < s2)):
                optical_correct += 1
print("temporal only: "+str(optical_correct/optical_count))

fusion_list = []
for i in range(0,28):
    fusion = 0.8*rgb_list[i] + 0.2*optical_list[i]
    fusion_list.append(fusion)

fusion_count = 0
fusion_correct = 0
for v in range(0,28):
    for k in range(v,28):
        s1 = rgb_list[math.floor(v/4)]
        s2 = rgb_list[math.floor(k/4)]
        if(year[math.floor(v/4)] != year[math.floor(k/4)]):
            fusion_count += 1
            if((year[math.floor(v/4)] > year[math.floor(k/4)] and s1 > s2)
            or (year[math.floor(v/4)] < year[math.floor(k/4)] and s1 < s2)):
                fusion_correct += 1
print("fusion: "+str(fusion_correct/fusion_count))
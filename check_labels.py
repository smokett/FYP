for folder in range(0,7):
    for label_file in range(1,21):
        file_name = ""
        file_name = str(folder)+'/'+str(label_file)+'.txt'
        f = open(file_name,'r')
        line = f.readline()
        num = len(line.split(','))
        if(num != 32):
            print("Checking " + str(folder)+'/'+str(label_file))
            print('1st ' + str(num))
        line = f.readline()
        num = len(line.split(','))
        if(num != 19):
            print("Checking " + str(folder)+'/'+str(label_file))
            print('2nd ' + str(num))
        f.close()
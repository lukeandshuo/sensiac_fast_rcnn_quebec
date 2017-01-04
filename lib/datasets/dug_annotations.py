import cv2
import os
data_path= os.path.join(os.path.dirname(__file__),'..','..','data','sample_data')
filename = os.path.join(data_path, 'Annotations', 'Visible', 'train.txt')
# print 'Loading: {}'.format(filename)
image_set_file = os.path.join(data_path, 'Train_Test','Visible',
                               'train.txt')
image_dir = os.path.join(data_path, 'Imagery','Visible','images')
image_index=[]
assert os.path.exists(image_set_file), \
    'Path does not exist: {}'.format(image_set_file)
with open(image_set_file) as f:
    image_index = [x.strip() for x in f.readlines()]

with open(filename) as f:
    for ind,line in enumerate(f):
        line = line.strip().split(",")
        num_objs = 1
        for i in range(num_objs):
            x1 = float(line[0 + i * 4])-5
            y1 = float(line[1 + i * 4])-5
            x2 = float(line[2 + i * 4])+5
            y2 = float(line[3 + i * 4])+5
            im = cv2.imread(os.path.join(image_dir,image_index[ind]+'.png'))
            cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0))
            cv2.imshow("dd",im)
            cv2.waitKey(24)
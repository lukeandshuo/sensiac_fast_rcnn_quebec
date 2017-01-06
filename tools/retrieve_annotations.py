import os
import os.path as op
import pandas
from pandas import Series, DataFrame
import numpy as np

agt = DataFrame.from_csv("FullList.csv")
test =[]
test_annotations = []
with open("test.txt",'r') as f:
    for l in f:
        l=l.strip('\n')
        test.append(l)
for l in test:
    record = agt[agt['ImageName']==l]
    x1 = record['UpperLeft'][0]
    y1 = record['UpperTop'][0]

    x2 = x1 + record['BBoxW'][0]
    y2 = y1 + record['BBoxH'][0]
    test_annotations.append("{}\t{}\t{}\t{}\n".format(x1,y1,x2,y2))
with open("test_annotation.txt",'w') as f:
    for line in test_annotations:
        f.write(line)


train =[]
train_annotations = []
with open("train.txt","r") as f:
    for l in f:
        l = l.strip('\n')
        train.append(l)
for l in train:
    record = agt[agt['ImageName']==l]
    x1 = record['UpperLeft'][0]
    y1 = record['UpperTop'][0]

    x2 = x1 + record['BBoxW'][0]
    y2 = y1 + record['BBoxH'][0]
    train_annotations.append("{}\t{}\t{}\t{}\n".format(x1,y1,x2,y2))
with open("train_annotation.txt",'w') as f:
    for line in train_annotations:
        f.write(line)
import cv2
import numpy as np
import os.path as op

feature_folder = "feature"
results_folder = "results"

for i in range(2400):
    feature = cv2.imread(op.join(feature_folder,str(i)+".png"))
    feature = cv2.resize(feature,(640,480))
    results = cv2.imread(op.join(results_folder,str(i)+".png"))
    # gap = np.ones((480,40,3))
    # print results.shape
    sum = np.concatenate((feature,results),axis=1)

    cv2.imshow("feature",sum)
    cv2.waitKey(30)
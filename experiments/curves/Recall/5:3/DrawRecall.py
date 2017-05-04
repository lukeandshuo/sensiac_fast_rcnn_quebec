import cPickle
import matplotlib.pyplot as plt
import os
import os.path as op
base_path= os.path.split(os.path.abspath(__file__))[0]
xlabel=[0.5,0.6,0.7,0.8,0.9]
IR=[0.429,0.196,0.044,0.001,0.00]
Visible =[0.850,0.615,0.320,0.090,0.005]
plt.plot(xlabel,Visible,'r',label='Visible =0.3329')
plt.plot(xlabel,IR,'b',label='IR = 0.1089')
plt.xlabel('IOU')
plt.ylabel('Recall')
plt.xlim([0.5,0.9])
plt.ylim([0.0,1.0])
plt.legend(loc='upper right')
plt.savefig('recall.png')
plt.show()
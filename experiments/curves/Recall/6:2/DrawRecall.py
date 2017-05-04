import cPickle
import matplotlib.pyplot as plt
import os
import os.path as op
base_path= os.path.split(os.path.abspath(__file__))[0]
xlabel=[0.5,0.6,0.7,0.8,0.9]
IR=[0.469,0.231,0.050,0.001,0.00]
Visible =[0.919,0.692,0.371,0.108,0.005]
plt.plot(xlabel,Visible,'r',label='Visible =0.3731')
plt.plot(xlabel,IR,'b',label='IR = 0.124')
plt.xlabel('IOU')
plt.ylabel('Recall')
plt.xlim([0.5,0.9])
plt.ylim([0.0,1.0])
plt.legend(loc='upper right')
plt.savefig('recall.png')
plt.show()
import cPickle
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
# from pandas import DataFrame as DF
# from ggplot import *
def concatenate(df):
    df['rec'] = np.concatenate(([0.0], df['rec'], [1.0]), axis=0)
    df['prec'] = np.concatenate(([1.0], df['prec'], [0.0]), axis=0)
    return df

def get_ap(type):
    dir = op.abspath(op.join("..","..","..","output",str(type),"test","vgg_cnn_m_1024_fast_rcnn_iter_40000","vehicle_ap.pkl"))
    with open(dir, 'r') as f:
        ap = concatenate(cPickle.load(f))
    return ap


# fig = plt.figure()
# ax= fig.gca()
# ax.set_xticks(np.arange(0,1,0.1))
# ax.set_yticks(np.arange(0,1,0.1))
lw = 2
v = get_ap("Visible")
plt.plot(v['rec'],v['prec'],'b--',label='[{:.4} %]:Visible '.format(v['ap']*100),linewidth=lw)
ir = get_ap("IR_Reg")
plt.plot(ir['rec'],ir['prec'],'r--',label='[{:.4} %]:MWIR '.format(ir['ap']*100),linewidth=lw)
m = get_ap("V_Motion")
plt.plot(m['rec'],m['prec'],'g--',label='[{:.4} %]:Motion '.format(m['ap']*100),linewidth=lw)
vir = get_ap("V_IR")
plt.plot(vir['rec'],vir['prec'],'k-',label='[{:.4} %]:Visible-MWIR '.format(vir['ap']*100),linewidth=lw)
threeC = get_ap("3C")
plt.plot(threeC['rec'],threeC['prec'],'y-',label='[{:.4} %]:3-Channels '.format(threeC['ap']*100),linewidth=lw)
ensemble = get_ap("Ensemble")
plt.plot(ensemble['rec'],ensemble['prec'],'m-',label='[{:.4} %]:Decision-level Fusion '.format(ensemble['ap']*100),linewidth=lw)
# threeC_EB = get_ap("3C_EB")
# plt.plot(threeC_EB['rec'],threeC_EB['prec'],'c--',label='[{:.4} %]:3C with EdgeBoxes '.format(threeC_EB['ap']*100),linewidth=lw)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True,linestyle='--')
plt.legend(loc='lower left')
plt.savefig('AP.png')
plt.show()

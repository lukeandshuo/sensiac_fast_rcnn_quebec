import cPickle
import numpy as np
import os
import os.path as op

def define_dir(type):
    return op.abspath(op.join("..","..","..","output",str(type),"test","vgg_cnn_m_1024_fast_rcnn_iter_40000","detections.pkl"))

IR_det_dir = define_dir("IR_Reg")
with open(IR_det_dir,"r") as f:
    ir_det = (cPickle.load(f))


V_det_dir = define_dir("Visible")
with open(V_det_dir,"r") as f:
    v_det = (cPickle.load(f))

M_det_dir =define_dir("V_Motion")
with open(M_det_dir,"r") as f:
    m_det = (cPickle.load(f))

image_size = 2812
c = 1
all = [[[]for _ in xrange(image_size)] for _ in xrange(2)]


print len(ir_det[1][1]),len(v_det[1][1])

for i in xrange(image_size):
    print i
    for j in xrange(len(ir_det[1][i])):
        all[c][i].append(ir_det[c][i][j])
    for j in xrange(len(v_det[1][i])):
        all[c][i].append(v_det[c][i][j])
    for j in xrange(len(m_det[1][i])):
        all[c][i].append(m_det[c][i][j])
    all[c][i] = np.array(all[c][i])
    print type(all[c][i])
result_dir = define_dir("Ensemble")
with open(result_dir,'w') as f:
    cPickle.dump(all,f,cPickle.HIGHEST_PROTOCOL)
import cPickle
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
base_path= os.path.split(os.path.abspath(__file__))[0]
IR_IR_V_dir = op.join(base_path,"M(IR)_I(IR)_R(V).pkl")
IR_IR_IR_dir = op.join(base_path,"M(IR)_I(IR)_R(IR).pkl")
IR_V_V_dir = op.join(base_path,"M(IR)_I(V)_R(V).pkl")
V_V_V_dir = op.join(base_path,"M(V)_I(V)_R(V).pkl")
V_V_VM_dir = op.join(base_path,"V_V_VM.pkl")
VM_VM_VM_dir = op.join(base_path,"VM_VM_VM.pkl")
def concatenate(df):
    df['rec'] = np.concatenate(([0.0], df['rec'], [1.0]), axis=0)
    df['prec'] = np.concatenate(([1.0], df['prec'], [0.0]), axis=0)
    return df
with open(IR_IR_V_dir,'r') as f:
    IR_IR_V = concatenate(cPickle.load(f))
with open(IR_IR_IR_dir,'r') as f:
    IR_IR_IR = concatenate(cPickle.load(f))
with open(IR_V_V_dir,'r') as f:
    IR_V_V = concatenate(cPickle.load(f))
with open(V_V_V_dir,'r') as f:
    V_V_V = concatenate(cPickle.load(f))
with open(V_V_VM_dir,'r') as f:
    V_V_VM = concatenate(cPickle.load(f))
with open(VM_VM_VM_dir,'r') as f:
    VM_VM_VM = concatenate(cPickle.load(f))

plt.plot(V_V_VM['rec'],V_V_VM['prec'],'g--',label='V_V_VM = {:.3}'.format(V_V_VM['ap']))
plt.plot(V_V_V['rec'],V_V_V['prec'],'g',label='V_V_V = {:.3}'.format(V_V_V['ap']))
plt.plot(VM_VM_VM['rec'],VM_VM_VM['prec'],'c',label='VM_VM_VM = {:.3}'.format(VM_VM_VM['ap']))
plt.plot(IR_V_V['rec'],IR_V_V['prec'],'r',label='IR_V_V = {:.3}'.format(IR_V_V['ap']))
plt.plot(IR_IR_V['rec'],IR_IR_V['prec'],'r--',label='IR_IR_V = {:.3}'.format(IR_IR_V['ap']))
plt.plot(IR_IR_IR['rec'],IR_IR_IR['prec'],'b',label='IR_IR_IR = {:.3}'.format(IR_IR_IR['ap']))


plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend(loc='lower left')
plt.savefig(op.join(base_path,'AP.png'))
plt.show()

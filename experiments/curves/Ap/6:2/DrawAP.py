import cPickle
import matplotlib.pyplot as plt
IR_IR_V_dir = "M(IR)_I(IR)_R(V)_pr.pkl"
IR_IR_IR_dir = "M(IR)_I(IR)_R(IR)_pr.pkl"
IR_V_V_dir = "M(IR)_I(V)_R(V)_pr.pkl"
V_V_V_dir = "M(V)_I(V)_R(V)_pr.pkl"
with open(IR_IR_V_dir,'r') as f:
    IR_IR_V = cPickle.load(f)
with open(IR_IR_IR_dir,'r') as f:
    IR_IR_IR = cPickle.load(f)
with open(IR_V_V_dir,'r') as f:
    IR_V_V = cPickle.load(f)
with open(V_V_V_dir,'r') as f:
    V_V_V = cPickle.load(f)

plt.plot(V_V_V['rec'],V_V_V['prec'],'g',label='V_V_V = {:.3}'.format(V_V_V['ap']))
plt.plot(IR_IR_IR['rec'],IR_IR_IR['prec'],'b',label='IR_IR_IR = {:.3}'.format(IR_IR_IR['ap']))
plt.plot(IR_V_V['rec'],IR_V_V['prec'],'r',label='IR_V_V = {:.3}'.format(IR_V_V['ap']))
plt.plot(IR_IR_V['rec'],IR_IR_V['prec'],'r--',label='IR_IR_V = {:.3}'.format(IR_IR_V['ap']))
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend(loc='lower left')
plt.savefig('AP.png')
plt.show()

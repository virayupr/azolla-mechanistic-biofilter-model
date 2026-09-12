import numpy as np
import pandas as pd
import os
import sobol_converged_2048 as m
from numba import njit

@njit(cache=True)
def room_trace(A,L,I,a_s,ACH,Qb,Pmax,alpha,Kc,Lc,dt=30.):
    V=m.V; COUT=m.COUT; C0=m.C0; TEND=m.TEND
    c=m.ppm2mol(C0); co=m.ppm2mol(COUT); Qv=ACH*V/3600.; n=int(TEND/dt)
    out=np.empty(n)
    for t in range(n):
        ppm=max(m.mol2ppm(c),COUT); cp,_=m.panel_out(ppm,Qb,A,L,I,a_s,Pmax,alpha,Kc,Lc); k1=(Qv*(co-c)+Qb*(m.ppm2mol(cp)-c))/V
        c2=c+.5*dt*k1; ppm=max(m.mol2ppm(c2),COUT); cp,_=m.panel_out(ppm,Qb,A,L,I,a_s,Pmax,alpha,Kc,Lc); k2=(Qv*(co-c2)+Qb*(m.ppm2mol(cp)-c2))/V
        c3=c+.5*dt*k2; ppm=max(m.mol2ppm(c3),COUT); cp,_=m.panel_out(ppm,Qb,A,L,I,a_s,Pmax,alpha,Kc,Lc); k3=(Qv*(co-c3)+Qb*(m.ppm2mol(cp)-c3))/V
        c4=c+dt*k3; ppm=max(m.mol2ppm(c4),COUT); cp,_=m.panel_out(ppm,Qb,A,L,I,a_s,Pmax,alpha,Kc,Lc); k4=(Qv*(co-c4)+Qb*(m.ppm2mol(cp)-c4))/V
        c += dt*(k1+2*k2+2*k3+k4)/6
        if c<co: c=co
        out[t]=m.mol2ppm(c)
    return out

vals=np.array([9.,.035,m.KC0,120.,.01,.18,.05,.02,500.,2.])
names=m.names

def trace_from(v):
    Pm,al,Kc,asv,Lc,A,L,Qb,I,ACH=v
    return room_trace(A,L,I,asv,ACH,Qb,Pm,al,Kc,Lc)

y0=trace_from(vals)
S=np.zeros((len(y0),len(vals)))
for j in range(len(vals)):
    h=0.01*abs(vals[j]) if vals[j]!=0 else 1e-6
    vp=vals.copy(); vm=vals.copy(); vp[j]+=h; vm[j]-=h
    yp=trace_from(vp); ym=trace_from(vm)
    dy=(yp-ym)/(2*h)
    scale=np.maximum(np.abs(y0-m.COUT),1.0)
    S[:,j]=dy*vals[j]/scale

corr=np.corrcoef(S,rowvar=False)
U,s,VT=np.linalg.svd(S,full_matrices=False)
od='results'; os.makedirs(od,exist_ok=True)
pd.DataFrame(corr,index=names,columns=names).to_csv(od+'/local_sensitivity_parameter_correlation.csv')
pd.DataFrame({'mode':np.arange(1,len(s)+1),'singular_value':s,'relative_to_max':s/s[0]}).to_csv(od+'/svd_singular_values.csv',index=False)
pd.DataFrame(S,columns=names).to_csv(od+'/normalized_local_sensitivity_matrix.csv',index=False)
print('corr a_s,A,L')
for a in ['a_s','A','L']:
    print(a, {b: corr[names.index(a),names.index(b)] for b in ['a_s','A','L']})
print('singular values',s)
print('relative',s/s[0])

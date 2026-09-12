import numpy as np, pandas as pd, time, os, math
from scipy.optimize import brentq
from scipy.stats import qmc
from numba import njit, prange

V=36.; C0=600.; COUT=400.; TEND=1800.; P=101325.; R=8.314462618; TK=298.15; MU=1.85e-5; RHO=1.184; DCO2=1.6e-5; CAIR=P/(R*TK)
KC0=brentq(lambda K:(640/(K+640))/(340/(K+340))-1.70,1e-6,1e7)

@njit(cache=True)
def ppm2mol(x): return x*1e-6*CAIR
@njit(cache=True)
def mol2ppm(x): return x/CAIR*1e6
@njit(cache=True)
def photo(I,alpha,Pmax):
    theta=.8; Rd=.5; X=alpha*I+Pmax; disc=X*X-4*theta*alpha*I*Pmax
    if disc<0: disc=0
    val=(X-math.sqrt(disc))/(2*theta)-Rd
    return val if val>0 else 0.0
@njit(cache=True)
def fco2(ppm,Kc): return ppm/(Kc+ppm)
@njit(cache=True)
def kg_external(u,Lc):
    Re=RHO*u*Lc/MU; Sc=(MU/RHO)/DCO2; Sh=.664*math.sqrt(max(Re,1e-12))*Sc**(1/3); return Sh*DCO2/Lc
@njit(cache=True)
def flux(ppm,u,I,Pmax,alpha,Kc,Lc):
    Ccomp=40.; kg=kg_external(u,Lc); d=ppm2mol(ppm)-ppm2mol(Ccomp); Jmt=kg*(d if d>0 else 0.0); Jbio=photo(I,alpha,Pmax)*(fco2(ppm,Kc)/fco2(400.,Kc))*1e-6; e=1e-30
    a=Jmt if Jmt>e else e; b=Jbio if Jbio>e else e
    return 1.0/(1.0/a+1.0/b)
@njit(cache=True)
def panel_out(Cin,Qb,A,L,I,a_s,Pmax,alpha,Kc,Lc,nx=40):
    u=Qb/A; c=ppm2mol(Cin); dx=L/nx; floor=ppm2mol(COUT)
    for zz in range(nx):
        ppm=max(mol2ppm(c),COUT); k1=-a_s*flux(ppm,u,I,Pmax,alpha,Kc,Lc)/max(u,1e-12)
        c2=c+.5*dx*k1; ppm=max(mol2ppm(c2),COUT); k2=-a_s*flux(ppm,u,I,Pmax,alpha,Kc,Lc)/max(u,1e-12)
        c3=c+.5*dx*k2; ppm=max(mol2ppm(c3),COUT); k3=-a_s*flux(ppm,u,I,Pmax,alpha,Kc,Lc)/max(u,1e-12)
        c4=c+dx*k3; ppm=max(mol2ppm(c4),COUT); k4=-a_s*flux(ppm,u,I,Pmax,alpha,Kc,Lc)/max(u,1e-12)
        c += dx*(k1+2*k2+2*k3+k4)/6
        if c<floor: c=floor
    Coutp=mol2ppm(c); den=Cin-COUT
    eta=(Cin-Coutp)/(den if den>1e-12 else 1e-12)
    if eta<0: eta=0.
    return Coutp,eta
@njit(cache=True)
def room_C30(A,L,I,a_s,ACH,Qb,Pmax,alpha,Kc,Lc,dt=30.):
    c=ppm2mol(C0); co=ppm2mol(COUT); Qv=ACH*V/3600.; floor=co
    n=int(TEND/dt)
    for t in range(n):
        ppm=max(mol2ppm(c),COUT); cp,_=panel_out(ppm,Qb,A,L,I,a_s,Pmax,alpha,Kc,Lc); k1=(Qv*(co-c)+Qb*(ppm2mol(cp)-c))/V
        c2=c+.5*dt*k1; ppm=max(mol2ppm(c2),COUT); cp,_=panel_out(ppm,Qb,A,L,I,a_s,Pmax,alpha,Kc,Lc); k2=(Qv*(co-c2)+Qb*(ppm2mol(cp)-c2))/V
        c3=c+.5*dt*k2; ppm=max(mol2ppm(c3),COUT); cp,_=panel_out(ppm,Qb,A,L,I,a_s,Pmax,alpha,Kc,Lc); k3=(Qv*(co-c3)+Qb*(ppm2mol(cp)-c3))/V
        c4=c+dt*k3; ppm=max(mol2ppm(c4),COUT); cp,_=panel_out(ppm,Qb,A,L,I,a_s,Pmax,alpha,Kc,Lc); k4=(Qv*(co-c4)+Qb*(ppm2mol(cp)-c4))/V
        c += dt*(k1+2*k2+2*k3+k4)/6
        if c<floor: c=floor
    return mol2ppm(c)
@njit(parallel=True,cache=True)
def batch_eval(X):
    n=X.shape[0]; out=np.empty((n,3))
    for r in prange(n):
        Pm,al,Kc,asv,Lc,A,L,Qb,I,ACH=X[r]
        c=room_C30(A,L,I,asv,ACH,Qb,Pm,al,Kc,Lc)
        _,e=panel_out(600.,Qb,A,L,I,asv,Pm,al,Kc,Lc)
        out[r,0]=c; out[r,1]=e; out[r,2]=Qb*e*3600.
    return out

names=['Pmax','alpha','Kc','a_s','Lc','A','L','Qb','I','ACH']
bounds=np.array([[2,12],[.015,.06],[1200,5000],[20,500],[.005,.03],[.05,.30],[.02,.10],[.005,.04],[100,850],[1,4]],float)
def scale(U): return bounds[:,0]+U*(bounds[:,1]-bounds[:,0])
def indices(YA,YB,YAB,seed,Bboot=1000):
    y=np.r_[YA,YB]; VY=np.var(y,ddof=1); D=YAB.shape[1]; rng=np.random.default_rng(seed)
    S1=np.empty(D); ST=np.empty(D); S1c=np.empty(D); STc=np.empty(D)
    n=len(YA)
    for i in range(D):
        S1[i]=np.mean(YB*(YAB[:,i]-YA))/VY; ST[i]=0.5*np.mean((YA-YAB[:,i])**2)/VY
        bs1=np.empty(Bboot); bst=np.empty(Bboot)
        for b in range(Bboot):
            ix=rng.integers(0,n,n); yy=np.r_[YA[ix],YB[ix]]; vv=np.var(yy,ddof=1)
            bs1[b]=np.mean(YB[ix]*(YAB[ix,i]-YA[ix]))/vv; bst[b]=0.5*np.mean((YA[ix]-YAB[ix,i])**2)/vv
        S1c[i]=1.96*np.std(bs1,ddof=1); STc[i]=1.96*np.std(bst,ddof=1)
    return S1,S1c,ST,STc

def main(Nmax=2048):
    D=len(names); sampler=qmc.Sobol(d=2*D,scramble=True,seed=20260911); U=sampler.random_base2(int(np.log2(Nmax))); A=scale(U[:,:D]); B=scale(U[:,D:]); mats=[A,B]
    for i in range(D):
        M=A.copy(); M[:,i]=B[:,i]; mats.append(M)
    X=np.vstack(mats); print('rows',len(X),flush=True)
    batch_eval(X[:2])
    t=time.time(); Y=batch_eval(X); print('eval seconds',time.time()-t,flush=True)
    Y=Y.reshape(D+2,Nmax,3); out=[]
    for N in [128,256,512,1024,2048]:
        for j,label in enumerate(['C30_ppm','eta_sp','CADR_m3h']):
            YA=Y[0,:N,j]; YB=Y[1,:N,j]; YAB=np.stack([Y[2+i,:N,j] for i in range(D)],axis=1)
            S1,S1c,ST,STc=indices(YA,YB,YAB,1000+N*10+j,1000)
            for k,p in enumerate(names): out.append([N,label,p,S1[k],S1c[k],ST[k],STc[k]])
    od='results'; os.makedirs(od,exist_ok=True)
    pd.DataFrame(out,columns=['N','output','parameter','S1','S1_conf','ST','ST_conf']).to_csv(od+'/sobol_convergence_128_2048.csv',index=False)
    np.savez_compressed(od+'/sobol_raw_2048.npz',Y=Y,U=U,bounds=bounds,names=np.array(names))

if __name__=='__main__':
    main()

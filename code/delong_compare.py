import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from canonical_analysis import load_data, build_models, RANDOM_STATE, OUT

def compute_midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1
        i=j
    T2=np.empty(N); T2[J]=T
    return T2

def fast_delong(predictions_sorted_transposed,label_1_count):
    m=label_1_count; n=predictions_sorted_transposed.shape[1]-m; k=predictions_sorted_transposed.shape[0]
    pos=predictions_sorted_transposed[:,:m]; neg=predictions_sorted_transposed[:,m:]
    tx=np.empty((k,m)); ty=np.empty((k,n)); tz=np.empty((k,m+n))
    for r in range(k):
        tx[r]=compute_midrank(pos[r]); ty[r]=compute_midrank(neg[r]); tz[r]=compute_midrank(predictions_sorted_transposed[r])
    aucs=tz[:,:m].sum(axis=1)/(m*n) - (m+1)/(2*n)
    v01=(tz[:,:m]-tx)/n; v10=1-(tz[:,m:]-ty)/m
    sx=np.cov(v01); sy=np.cov(v10); s=sx/m+sy/n
    return aucs,s

def delong_pvalue(y_true,p1,p2):
    y=np.asarray(y_true); order=np.argsort(-y); m=int(y.sum())
    preds=np.vstack([p1,p2])[:,order]
    aucs,cov=fast_delong(preds,m)
    l=np.array([[1,-1]])
    var=float(l@cov@l.T)
    if var<=0: return aucs[0],aucs[1],1.0
    z=abs(aucs[0]-aucs[1])/np.sqrt(var)
    p=2*norm.sf(z)
    return aucs[0],aucs[1],p

def holm(ps):
    from statsmodels.stats.multitest import multipletests
    return multipletests(ps,method='holm')[1]

X,y,_=load_data(); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=RANDOM_STATE)
mods=build_models(); probs={}
base=['Logistic Regression','SVM (RBF)','Random Forest','XGBoost']
arr=[]
for name in base:
    m=clone(mods[name]).fit(Xtr,ytr); probs[name]=m.predict_proba(Xte)[:,1]; arr.append(probs[name])
probs['Soft Voting Ensemble']=np.mean(np.vstack(arr),axis=0)
ref='Random Forest'; rows=[]; ps=[]
for name,p in probs.items():
    if name==ref: continue
    a1,a2,pv=delong_pvalue(yte,probs[ref],p); rows.append({'Reference':ref,'Comparator':name,'Reference AUC':a1,'Comparator AUC':a2,'DeLong p':pv}); ps.append(pv)
adj=holm(ps)
for r,a in zip(rows,adj): r['Holm-adjusted p']=a
out=pd.DataFrame(rows); out.to_csv(OUT/'pairwise_delong_vs_random_forest.csv',index=False); print(out.to_string(index=False))

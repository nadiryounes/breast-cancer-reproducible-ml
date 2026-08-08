from __future__ import annotations
import json, platform, sys, time, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss, average_precision_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier
import sklearn, scipy, matplotlib, xgboost

warnings.filterwarnings('ignore')
RANDOM_STATE=42
TEST_SIZE=0.20
CV_SPLITS=5
CV_REPEATS=3
BOOTSTRAPS=5000
STRESS_SIZES=[5_000,20_000,50_000,100_000,200_000]
STRESS_REPEATS=3
OUT=Path(__file__).resolve().parent / 'generated_outputs'
OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    ds=load_breast_cancer(as_frame=True)
    X=ds.data.copy()
    y=(ds.target==0).astype(int) # malignant positive
    return X,y,ds


def build_models():
    lr=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000,random_state=RANDOM_STATE))])
    svm=Pipeline([('scaler',StandardScaler()),('model',SVC(kernel='rbf',probability=True,random_state=RANDOM_STATE))])
    rf=RandomForestClassifier(n_estimators=400,random_state=RANDOM_STATE,n_jobs=1,class_weight='balanced')
    xgb=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,
                      objective='binary:logistic',eval_metric='logloss',random_state=RANDOM_STATE,n_jobs=1,tree_method='hist')
    ens=VotingClassifier(estimators=[('lr',lr),('svm',svm),('rf',rf),('xgb',xgb)],voting='soft',n_jobs=1)
    return {'Logistic Regression':lr,'SVM (RBF)':svm,'Random Forest':rf,'XGBoost':xgb,'Soft Voting Ensemble':ens}


def derived_metrics(y,pred,prob):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    spec=tn/(tn+fp) if tn+fp else np.nan
    npv=tn/(tn+fn) if tn+fn else np.nan
    return {
        'Accuracy':accuracy_score(y,pred),
        'Precision':precision_score(y,pred,zero_division=0),
        'Sensitivity (Recall)':recall_score(y,pred,zero_division=0),
        'Specificity':spec,
        'NPV':npv,
        'F1':f1_score(y,pred,zero_division=0),
        'Balanced Accuracy':balanced_accuracy_score(y,pred),
        'MCC':matthews_corrcoef(y,pred),
        'ROC-AUC':roc_auc_score(y,prob),
        'PR-AUC':average_precision_score(y,prob),
        'Brier Score':brier_score_loss(y,prob),
        'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)
    }


def bootstrap_ci(y,pred,prob,n=BOOTSTRAPS,seed=RANDOM_STATE):
    """Vectorized nonparametric bootstrap on the fixed held-out sample."""
    rng=np.random.default_rng(seed); y=np.asarray(y,int); pred=np.asarray(pred,int); prob=np.asarray(prob,float); N=len(y)
    idx=rng.integers(0,N,size=(n,N))
    counts=np.zeros((n,N),dtype=np.int16)
    np.add.at(counts,(np.repeat(np.arange(n),N),idx.ravel()),1)
    masks={
        'tp':((y==1)&(pred==1)).astype(int),'tn':((y==0)&(pred==0)).astype(int),
        'fp':((y==0)&(pred==1)).astype(int),'fn':((y==1)&(pred==0)).astype(int)
    }
    tp=counts@masks['tp']; tn=counts@masks['tn']; fp=counts@masks['fp']; fn=counts@masks['fn']
    npos=tp+fn; nneg=tn+fp; valid=(npos>0)&(nneg>0)
    acc=(tp+tn)/N
    sens=np.divide(tp,npos,out=np.full(n,float('nan')),where=npos>0)
    spec=np.divide(tn,nneg,out=np.full(n,float('nan')),where=nneg>0)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(n,float),where=(2*tp+fp+fn)>0)
    bal=(sens+spec)/2
    denom=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)).astype(float)
    mcc=np.divide(tp*tn-fp*fn,denom,out=np.zeros(n,float),where=denom>0)
    brier=(counts@((prob-y)**2))/N
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    cmp=(prob[pos,None]>prob[neg][None,:]).astype(float)+0.5*(prob[pos,None]==prob[neg][None,:])
    auc_num=np.einsum('bi,ij,bj->b',counts[:,pos].astype(float),cmp,counts[:,neg].astype(float),optimize=True)
    auc=np.divide(auc_num,npos*nneg,out=np.full(n,float('nan')),where=valid)
    vals={'Accuracy':acc,'Sensitivity (Recall)':sens,'Specificity':spec,'F1':f1,'Balanced Accuracy':bal,'MCC':mcc,'ROC-AUC':auc,'Brier Score':brier}
    return {k:(float(np.nanquantile(v[valid],.025)),float(np.nanquantile(v[valid],.975))) for k,v in vals.items()}


def holm_adjust(pvalues):
    return multipletests(pvalues, method='holm')[1]


def evaluate(models,Xtr,ytr,Xte,yte):
    cv=RepeatedStratifiedKFold(n_splits=CV_SPLITS,n_repeats=CV_REPEATS,random_state=RANDOM_STATE)
    base_names=['Logistic Regression','SVM (RBF)','Random Forest','XGBoost']
    cv_fold_rows=[]
    for fold,(itr,iva) in enumerate(cv.split(Xtr,ytr),start=1):
        Xa,Xv=Xtr.iloc[itr],Xtr.iloc[iva]; ya,yv=ytr.iloc[itr],ytr.iloc[iva]
        fold_probs=[]
        for name in base_names:
            m=clone(models[name]).fit(Xa,ya)
            pr=m.predict_proba(Xv)[:,1]; pd_=(pr>=0.5).astype(int); fold_probs.append(pr)
            cv_fold_rows.append({'Model':name,'Fold':fold,'CV F1':f1_score(yv,pd_,zero_division=0),'CV ROC-AUC':roc_auc_score(yv,pr)})
        ens_pr=np.mean(np.vstack(fold_probs),axis=0); ens_pd=(ens_pr>=0.5).astype(int)
        cv_fold_rows.append({'Model':'Soft Voting Ensemble','Fold':fold,'CV F1':f1_score(yv,ens_pd,zero_division=0),'CV ROC-AUC':roc_auc_score(yv,ens_pr)})
    cvdf=pd.DataFrame(cv_fold_rows); cvdf.to_csv(OUT/'repeated_cv_fold_results.csv',index=False)
    cvsum=cvdf.groupby('Model').agg(**{'Repeated CV ROC-AUC Mean':('CV ROC-AUC','mean'),'Repeated CV ROC-AUC SD':('CV ROC-AUC','std'),'Repeated CV F1 Mean':('CV F1','mean'),'Repeated CV F1 SD':('CV F1','std')}).reset_index().set_index('Model')

    perf=[]; details={}; final_probs=[]; fitted_base={}
    for name in base_names:
        fitted=clone(models[name]).fit(Xtr,ytr); fitted_base[name]=fitted
        pred=fitted.predict(Xte); prob=fitted.predict_proba(Xte)[:,1]; final_probs.append(prob)
        m=derived_metrics(yte,pred,prob); cis=bootstrap_ci(yte,pred,prob,seed=RANDOM_STATE+len(perf)*11)
        row={'Model':name,**cvsum.loc[name].to_dict(),**m}
        for metric,(lo,hi) in cis.items(): row[f'{metric} CI95 Low']=lo; row[f'{metric} CI95 High']=hi
        perf.append(row); details[name]={'model':fitted,'pred':pred,'prob':prob,'cm':confusion_matrix(yte,pred,labels=[0,1])}
    ens_prob=np.mean(np.vstack(final_probs),axis=0); ens_pred=(ens_prob>=0.5).astype(int)
    m=derived_metrics(yte,ens_pred,ens_prob); cis=bootstrap_ci(yte,ens_pred,ens_prob,seed=RANDOM_STATE+44)
    row={'Model':'Soft Voting Ensemble',**cvsum.loc['Soft Voting Ensemble'].to_dict(),**m}
    for metric,(lo,hi) in cis.items(): row[f'{metric} CI95 Low']=lo; row[f'{metric} CI95 High']=hi
    perf.append(row); details['Soft Voting Ensemble']={'model':None,'base_models':fitted_base,'pred':ens_pred,'prob':ens_prob,'cm':confusion_matrix(yte,ens_pred,labels=[0,1])}
    df=pd.DataFrame(perf).sort_values(['ROC-AUC','Balanced Accuracy','F1'],ascending=False).reset_index(drop=True)
    df.to_csv(OUT/'model_performance_table.csv',index=False); return df,details


def pairwise_prediction_tests(details,yte,reference='Random Forest'):
    ref=details[reference]['pred']; rows=[]
    pvals=[]
    for name,d in details.items():
        if name==reference: continue
        pred=d['pred']
        ref_correct=(ref==np.asarray(yte)); other_correct=(pred==np.asarray(yte))
        b=int(np.sum(ref_correct & ~other_correct)); c=int(np.sum(~ref_correct & other_correct))
        if b+c:
            p=binomtest(min(b,c),n=b+c,p=.5,alternative='two-sided').pvalue
        else: p=1.0
        rows.append({'Reference':reference,'Comparator':name,'Ref correct / comparator wrong':b,'Ref wrong / comparator correct':c,'McNemar exact p':p})
        pvals.append(p)
    adj=holm_adjust(pvals)
    for r,a in zip(rows,adj): r['Holm-adjusted p']=float(a)
    out=pd.DataFrame(rows)
    out.to_csv(OUT/'pairwise_mcnemar_vs_random_forest.csv',index=False)
    return out


def plot_model_comparison(df):
    order=list(df['Model'])
    vals=df.set_index('Model').loc[order,['Accuracy','F1','ROC-AUC','Balanced Accuracy']]
    ax=vals.plot(kind='bar',figsize=(10,5.8),rot=18)
    ax.set_ylabel('Score'); ax.set_ylim(.85,1.01); ax.set_title('Held-out test performance')
    ax.legend(loc='lower left',ncol=2,fontsize=8); plt.tight_layout(); plt.savefig(OUT/'Fig2_model_comparison.png',dpi=300); plt.close()


def plot_roc(details,yte):
    plt.figure(figsize=(7.6,6))
    for name,d in details.items():
        fpr,tpr,_=roc_curve(yte,d['prob']); auc=roc_auc_score(yte,d['prob'])
        plt.plot(fpr,tpr,label=f'{name} (AUC={auc:.4f})')
    plt.plot([0,1],[0,1],'--',linewidth=1)
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('ROC curves on the held-out test set')
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT/'Fig3_roc_curves.png',dpi=300); plt.close()


def plot_confusions(details):
    # Canonical presentation model: Soft Voting Ensemble, retained as prespecified integrative comparator, not declared best.
    d=details['Soft Voting Ensemble']
    fig,ax=plt.subplots(figsize=(5,4.4))
    ConfusionMatrixDisplay(d['cm'],display_labels=['Benign','Malignant']).plot(ax=ax,colorbar=False)
    ax.set_title('Soft-voting ensemble (threshold = 0.50)', fontsize=11)
    plt.tight_layout(); plt.savefig(OUT/'Fig4_ensemble_confusion_matrix.png',dpi=300); plt.close()


def plot_calibration(details,yte):
    rows=[]
    plt.figure(figsize=(7.6,6))
    for name,d in details.items():
        frac,mean=calibration_curve(yte,d['prob'],n_bins=8,strategy='quantile')
        plt.plot(mean,frac,marker='o',label=f"{name} (Brier={brier_score_loss(yte,d['prob']):.3f})")
        for m,f in zip(mean,frac): rows.append({'Model':name,'mean_predicted_probability':m,'fraction_positive':f})
    plt.plot([0,1],[0,1],'--',linewidth=1)
    plt.xlabel('Mean predicted probability'); plt.ylabel('Observed malignant fraction'); plt.title('Calibration curves on the held-out test set')
    plt.legend(fontsize=7.5); plt.tight_layout(); plt.savefig(OUT/'Fig5_calibration_curves.png',dpi=300); plt.close()
    pd.DataFrame(rows).to_csv(OUT/'calibration_curve_points.csv',index=False)


def explainability(details,Xte,yte):
    base_models=details['Soft Voting Ensemble']['base_models']
    baseline=roc_auc_score(yte,details['Soft Voting Ensemble']['prob'])
    rng=np.random.default_rng(RANDOM_STATE); rows=[]
    for col in Xte.columns:
        drops=[]
        for _ in range(20):
            Xp=Xte.copy(); Xp[col]=rng.permutation(Xp[col].to_numpy())
            probs=[m.predict_proba(Xp)[:,1] for m in base_models.values()]
            auc=roc_auc_score(yte,np.mean(np.vstack(probs),axis=0)); drops.append(baseline-auc)
        rows.append({'Feature':col,'Permutation Importance Mean':float(np.mean(drops)),'Permutation Importance SD':float(np.std(drops,ddof=1))})
    imp=pd.DataFrame(rows).sort_values('Permutation Importance Mean',ascending=False); imp.to_csv(OUT/'ensemble_permutation_importance.csv',index=False)
    top=imp.head(12).sort_values('Permutation Importance Mean')
    plt.figure(figsize=(8,5.4)); plt.barh(top['Feature'],top['Permutation Importance Mean'],xerr=top['Permutation Importance SD'])
    plt.xlabel('Decrease in ROC-AUC after permutation'); plt.title('Model-agnostic permutation importance: soft voting ensemble')
    plt.tight_layout(); plt.savefig(OUT/'Fig6_ensemble_permutation_importance.png',dpi=300); plt.close()

    # Native XGBoost TreeSHAP contributions avoid a separate SHAP-runtime dependency.
    from xgboost import DMatrix
    xgb=details['XGBoost']['model']
    contrib=xgb.get_booster().predict(DMatrix(Xte), pred_contribs=True)[:, :-1]
    mean_abs=np.abs(contrib).mean(axis=0)
    shap_imp=pd.DataFrame({'Feature':Xte.columns,'Mean absolute SHAP':mean_abs}).sort_values('Mean absolute SHAP',ascending=False)
    shap_imp.to_csv(OUT/'xgboost_shap_importance.csv',index=False)
    top_features=shap_imp.head(12)['Feature'].tolist()[::-1]
    fig,ax=plt.subplots(figsize=(8.2,6.2))
    jitter_rng=np.random.default_rng(RANDOM_STATE)
    sc=None
    for yi,feat in enumerate(top_features):
        j=Xte.columns.get_loc(feat); vals=Xte[feat].to_numpy(float); sv=contrib[:,j]
        lo,hi=np.nanmin(vals),np.nanmax(vals); norm=(vals-lo)/(hi-lo) if hi>lo else np.full_like(vals,.5)
        jitter=jitter_rng.normal(0,0.07,size=len(vals))
        sc=ax.scatter(sv, np.full(len(vals),yi)+jitter, c=norm, cmap='coolwarm', vmin=0, vmax=1, s=13, alpha=.85, linewidths=0)
    ax.axvline(0,linewidth=.8,color='0.45'); ax.set_yticks(range(len(top_features))); ax.set_yticklabels(top_features)
    ax.set_xlabel('SHAP value (impact on model output)'); ax.set_title('XGBoost TreeSHAP summary (held-out test set)')
    cbar=fig.colorbar(sc,ax=ax,pad=.02); cbar.set_label('Feature value'); cbar.set_ticks([0,1]); cbar.set_ticklabels(['Low','High'])
    fig.tight_layout(); fig.savefig(OUT/'Fig7_xgboost_shap_summary.png',dpi=300,bbox_inches='tight'); plt.close(fig)


def error_analysis(details,Xte,yte):
    idx=np.asarray(yte.index)
    rows=[]
    for name,d in details.items():
        pred=np.asarray(d['pred']); prob=np.asarray(d['prob']); yy=np.asarray(yte)
        mis=np.where(pred!=yy)[0]
        for j in mis:
            row={'Model':name,'Original row index':int(idx[j]),'True class':'Malignant' if yy[j]==1 else 'Benign',
                 'Predicted class':'Malignant' if pred[j]==1 else 'Benign','P(malignant)':float(prob[j])}
            for f in ['worst radius','worst perimeter','worst concave points','mean concave points','worst texture']:
                if f in Xte.columns: row[f]=float(Xte.iloc[j][f])
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'misclassified_cases.csv',index=False)


def synthetic_expand(X,y,n,seed,noise_scale=.01):
    rng=np.random.default_rng(seed)
    Xp=X[y==1]; Xn=X[y==0]; ratio=float((y==1).mean()); np_=int(round(n*ratio)); nn=n-np_
    a=Xp.iloc[rng.integers(0,len(Xp),np_)].reset_index(drop=True); b=Xn.iloc[rng.integers(0,len(Xn),nn)].reset_index(drop=True)
    Xs=pd.concat([a,b],ignore_index=True); ys=pd.Series(np.r_[np.ones(np_,int),np.zeros(nn,int)])
    ranges=(X.max()-X.min()).replace(0,1.0)
    Xs=Xs+rng.normal(0,noise_scale,size=Xs.shape)*ranges.to_numpy(); Xs=Xs.clip(lower=X.min(),upper=X.max(),axis=1)
    p=rng.permutation(n); return Xs.iloc[p].reset_index(drop=True),ys.iloc[p].reset_index(drop=True)


def stress_test(Xtr,ytr,Xte,yte):
    rows=[]
    for n in STRESS_SIZES:
        for r in range(STRESS_REPEATS):
            Xb,yb=synthetic_expand(Xtr,ytr,n,RANDOM_STATE+n+r*1000)
            models={
                'Logistic Regression':Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000,random_state=RANDOM_STATE))]),
                'XGBoost':XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,objective='binary:logistic',eval_metric='logloss',random_state=RANDOM_STATE,n_jobs=1,tree_method='hist')
            }
            for name,m in models.items():
                t0=time.perf_counter(); m.fit(Xb,yb); elapsed=time.perf_counter()-t0
                pr=m.predict_proba(Xte)[:,1]
                rows.append({'Model':name,'n_records':n,'Repeat':r+1,'training_time_seconds':elapsed,'test_roc_auc':roc_auc_score(yte,pr)})
    raw=pd.DataFrame(rows); raw.to_csv(OUT/'stress_test_raw.csv',index=False)
    agg=raw.groupby(['Model','n_records']).agg(training_time_mean=('training_time_seconds','mean'),training_time_sd=('training_time_seconds','std'),test_roc_auc_mean=('test_roc_auc','mean'),test_roc_auc_sd=('test_roc_auc','std')).reset_index()
    agg.to_csv(OUT/'stress_test_results.csv',index=False)
    plt.figure(figsize=(8,5.2))
    for name,g in agg.groupby('Model'):
        g=g.sort_values('n_records'); plt.errorbar(g['n_records'],g['training_time_mean'],yerr=g['training_time_sd'],marker='o',capsize=3,label=name)
    plt.xlabel('Synthetic training records'); plt.ylabel('Training time (s), mean ± SD'); plt.title('Computational stress test (3 repetitions)')
    plt.legend(); plt.tight_layout(); plt.savefig(OUT/'Fig8_stress_test_training_time.png',dpi=300); plt.close()
    plt.figure(figsize=(8,5.2))
    for name,g in agg.groupby('Model'):
        g=g.sort_values('n_records'); plt.errorbar(g['n_records'],g['test_roc_auc_mean'],yerr=g['test_roc_auc_sd'],marker='o',capsize=3,label=name)
    plt.xlabel('Synthetic training records'); plt.ylabel('Held-out ROC-AUC, mean ± SD'); plt.title('Discrimination stability during computational stress testing')
    plt.legend(); plt.tight_layout(); plt.savefig(OUT/'Supplementary_Fig_S1_stress_auc.png',dpi=300); plt.close()
    return agg


def plot_workflow():
    fig,ax=plt.subplots(figsize=(7.2,2.5)); ax.axis('off')
    boxes=[
        'WDBC\n569 × 30',
        '80/20 split\n5×3 repeated CV',
        'Held-out test\nbootstrap CIs + calibration',
        'Error analysis + XAI\nsynthetic stress test',
    ]
    xs=np.array([.10,.36,.64,.90]); y=.57
    for i,(txt,x) in enumerate(zip(boxes,xs)):
        ax.text(x,y,txt,ha='center',va='center',fontsize=9.5,
                bbox=dict(boxstyle='round,pad=.42',fc='white',ec='black'))
        if i<len(boxes)-1:
            ax.annotate('',xy=(xs[i+1]-.105,y),xytext=(x+.105,y),
                        arrowprops=dict(arrowstyle='->',lw=1.2))
    ax.text(.5,.15,'One version-pinned pipeline generates all reported numerical results, tables, and figures.',
            ha='center',fontsize=8.8)
    plt.tight_layout(); plt.savefig(OUT/'Fig1_workflow.png',dpi=300,bbox_inches='tight'); plt.close()


def save_env():
    import psutil
    env={
        'python':sys.version.split()[0], 'numpy':np.__version__, 'pandas':pd.__version__, 'scikit-learn':sklearn.__version__,
        'xgboost':xgboost.__version__, 'scipy':scipy.__version__, 'matplotlib':matplotlib.__version__,
        'platform':platform.platform(), 'machine':platform.machine(), 'logical_cpus':psutil.cpu_count(logical=True),
        'physical_cpus':psutil.cpu_count(logical=False),'ram_gib':round(psutil.virtual_memory().total/1024**3,2),
        'cpu_model': next((line.split(':',1)[1].strip() for line in open('/proc/cpuinfo') if line.startswith('model name')), 'unknown')
    }
    (OUT/'environment.json').write_text(json.dumps(env,indent=2))
    req=[f"numpy=={np.__version__}",f"pandas=={pd.__version__}",f"scikit-learn=={sklearn.__version__}",f"xgboost=={xgboost.__version__}",f"scipy=={scipy.__version__}",f"matplotlib=={matplotlib.__version__}","statsmodels==0.14.5","psutil==7.0.0"]
    (OUT/'requirements.txt').write_text('\n'.join(req)+'\n')


def main():
    save_env(); X,y,ds=load_data()
    dq=pd.DataFrame([{'n_samples':len(X),'n_features':X.shape[1],'missing_values_total':int(X.isna().sum().sum()),'duplicated_rows':int(X.duplicated().sum()),'malignant':int(y.sum()),'benign':int((1-y).sum())}]); dq.to_csv(OUT/'data_quality_report.csv',index=False)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    pd.DataFrame([{'Total cases':len(X),'Predictors':X.shape[1],'Benign':int((1-y).sum()),'Malignant':int(y.sum()),'Training cases':len(Xtr),'Test cases':len(Xte),'Training malignant':int(ytr.sum()),'Training benign':int((1-ytr).sum()),'Test malignant':int(yte.sum()),'Test benign':int((1-yte).sum())}]).to_csv(OUT/'dataset_description_table.csv',index=False)
    models=build_models(); perf,details=evaluate(models,Xtr,ytr,Xte,yte)
    pairwise_prediction_tests(details,yte)
    plot_workflow(); plot_model_comparison(perf); plot_roc(details,yte); plot_confusions(details); plot_calibration(details,yte); explainability(details,Xte,yte); error_analysis(details,Xte,yte); stress=stress_test(Xtr,ytr,Xte,yte)
    print(perf[['Model','Repeated CV ROC-AUC Mean','Repeated CV ROC-AUC SD','Accuracy','Sensitivity (Recall)','Specificity','F1','Balanced Accuracy','MCC','ROC-AUC','Brier Score','TN','FP','FN','TP']].to_string(index=False))
    print('\nStress\n',stress.to_string(index=False))
    print('\nSaved to',OUT)



def run_core_only():
    save_env(); X,y,_=load_data()
    dq=pd.DataFrame([{'n_samples':len(X),'n_features':X.shape[1],'missing_values_total':int(X.isna().sum().sum()),'duplicated_rows':int(X.duplicated().sum()),'malignant':int(y.sum()),'benign':int((1-y).sum())}]); dq.to_csv(OUT/'data_quality_report.csv',index=False)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    pd.DataFrame([{'Total cases':len(X),'Predictors':X.shape[1],'Benign':int((1-y).sum()),'Malignant':int(y.sum()),'Training cases':len(Xtr),'Test cases':len(Xte),'Training malignant':int(ytr.sum()),'Training benign':int((1-ytr).sum()),'Test malignant':int(yte.sum()),'Test benign':int((1-yte).sum())}]).to_csv(OUT/'dataset_description_table.csv',index=False)
    perf,details=evaluate(build_models(),Xtr,ytr,Xte,yte); pairwise_prediction_tests(details,yte)
    plot_workflow(); plot_model_comparison(perf); plot_roc(details,yte); plot_confusions(details); plot_calibration(details,yte); explainability(details,Xte,yte); error_analysis(details,Xte,yte)

def stress_one(n,r):
    X,y,_=load_data(); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    Xb,yb=synthetic_expand(Xtr,ytr,n,RANDOM_STATE+n+(r-1)*1000); rows=[]
    mods={'Logistic Regression':Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000,random_state=RANDOM_STATE))]),'XGBoost':XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,objective='binary:logistic',eval_metric='logloss',random_state=RANDOM_STATE,n_jobs=1,tree_method='hist')}
    for name,m in mods.items():
        t0=time.perf_counter(); m.fit(Xb,yb); elapsed=time.perf_counter()-t0; pr=m.predict_proba(Xte)[:,1]
        rows.append({'Model':name,'n_records':n,'Repeat':r,'training_time_seconds':elapsed,'test_roc_auc':roc_auc_score(yte,pr)})
    pd.DataFrame(rows).to_csv(OUT/f'stress_piece_r{r}_n{n}.csv',index=False)

def run_stress_all():
    X,y,_=load_data(); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    stress_test(Xtr,ytr,Xte,yte)

def aggregate_stress_pieces():
    parts=list(OUT.glob('stress_piece_r*_n*.csv'))
    raw=pd.concat([pd.read_csv(p) for p in parts],ignore_index=True).sort_values(['Model','n_records','Repeat']); raw.to_csv(OUT/'stress_test_raw.csv',index=False)
    agg=raw.groupby(['Model','n_records']).agg(training_time_mean=('training_time_seconds','mean'),training_time_sd=('training_time_seconds','std'),test_roc_auc_mean=('test_roc_auc','mean'),test_roc_auc_sd=('test_roc_auc','std')).reset_index(); agg.to_csv(OUT/'stress_test_results.csv',index=False)
    plt.figure(figsize=(8,5.2))
    for name,g in agg.groupby('Model'):
        g=g.sort_values('n_records'); plt.errorbar(g['n_records'],g['training_time_mean'],yerr=g['training_time_sd'],marker='o',capsize=3,label=name)
    plt.xlabel('Synthetic training records'); plt.ylabel('Training time (s), mean ± SD'); plt.title('Computational stress test (3 repetitions)'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'Fig8_stress_test_training_time.png',dpi=300); plt.close()
    plt.figure(figsize=(8,5.2))
    for name,g in agg.groupby('Model'):
        g=g.sort_values('n_records'); plt.errorbar(g['n_records'],g['test_roc_auc_mean'],yerr=g['test_roc_auc_sd'],marker='o',capsize=3,label=name)
    plt.xlabel('Synthetic training records'); plt.ylabel('Held-out ROC-AUC, mean ± SD'); plt.title('Discrimination stability during computational stress testing'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'Supplementary_Fig_S1_stress_auc.png',dpi=300); plt.close()

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--core',action='store_true'); ap.add_argument('--stress-all',action='store_true'); ap.add_argument('--stress-one',nargs=2,type=int,metavar=('N','REPEAT')); ap.add_argument('--aggregate-stress',action='store_true'); args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.core: run_core_only()
    elif args.stress_all: run_stress_all()
    elif args.stress_one: stress_one(*args.stress_one)
    elif args.aggregate_stress: aggregate_stress_pieces()
    else: ap.print_help()

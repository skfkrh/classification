# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 00:14:06 2020

@author: Min
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing 
from sklearn.cluster import KMeans

loc=r"C:\Users\Min\Downloads\takehome_ds_written.csv"
df=pd.read_csv(loc)
df.head()
df.columns
df.amount_usd_in_cents =round(df.amount_usd_in_cents/100)

df.dtypes
df.describe()

df.isnull().sum()

#perMerchant=df.groupby("merchant").size()

def perc1(sr):
    return np.percentiles(sr, np.linspace(0,100,11)) 

def perc2(sr):
    return np.percentile(sr, np.linspace(0,100,21)) 

#plt.scatter(range(perMerchant.shape[0]), perMerchant, sort_values()) 
#plt.ylim( 1,5000)

np.percentile(df.amount_usd_in_cents, np.linspace(95,100,20)) 


plt.hist(np.log(df.amount_usd_in_cents))


df.time=pd.to_datetime(df.time)
df['hour']=df.time.dt.hour
df['weekday' ]=df.time.dt.weekday 

#df['wknd']=df['weekday' ]<5
df['wknd']=df['weekday' ]>=5

df=df.sort_values(by=['merchant', 'time']) 
df['time_1']=df.time.shift(-1)
df['day']=df.time.dt.day

#characterize transaction
#tranlist=df.copy(deep=True)
def generatefeatures(tranlist, time_st, time_end):
    from scipy.stats import linregress
    eligible = (tranlist.time >time_st) & (tranlist.time<time_end)
    df=tranlist[eligible]
    
    df['merchant_1']=df.merchant.shift(-1)
    df['merchant_1_']=df.merchant.shift(1)
    
    df['tint']=(df.time_1-df.time).dt.days
    df['tint'][df.merchant != df.merchant_1] = None
    df['stt']= df.merchant != df.merchant_1_
    
    df["endt" ]=df.merchant != df.merchant_1
    df['timeindays']=(time_end - df.time).apply(lambda x: x.days).values
    
    df['tinactive']=np.nan 
    df['tinactive'][df.merchant != df.merchant_1]= df['timeindays'][df.merchant != df.merchant_1]
    df.tint[df.merchant != df.merchant_1]=None
    df['time2']=((df.time-time_st).dt.seconds+(df.time-time_st).dt.days*24*60*60)/3600
    dayAgg=df.groupby(['merchant','timeindays'])['amount_usd_in_cents'].sum().reset_index()
    
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_
    
    df_agg=df.groupby('merchant').agg({ 'amount_usd_in_cents':['mean','std',percentile(95),percentile(50),percentile(5)],
                                   'tint':['mean','std'],
                                   'day':['mean','std'],
                                   'wknd':['sum'],
                                   'merchant':['count'],
                                   'hour':['mean','std']})
    df_agg.columns=['_'.join(col).strip() for col in df_agg.columns.values]
    df_agg['trend']=df.groupby('merchant').apply(lambda v: linregress(v.time2, v.amount_usd_in_cents)[0])
    df_agg['trendtind']=df.dropna(subset=['tint']).groupby('merchant').apply(lambda v: linregress(v.time2, v.tint)[0]) 
    df_agg['trendDay']=dayAgg.groupby('merchant').apply(lambda v: linregress(v.timeindays, v.amount_usd_in_cents)[0]) 
    df_agg['endtime']=df.time[df.endt].values
    df_agg['length']= pd.Series(df.time[df.endt].values - df.time[df.stt].values).apply(lambda x: x.days).values
    df_agg['tinactive']=pd.Series(time_end-df.time[df.endt]).apply(lambda x: x.days).values
    
    
    bins=pd.IntervalIndex.from_tuples([(-1,15),(15,31)])
    df['dayCut']=pd.cut(df.day,bins)
    df['dayCut']=df['dayCut'].astype(str)
    
    bins=pd.IntervalIndex.from_tuples([(-1,8),(8,18),(18,23)])
    df['hourCut']=pd.cut(df.hour,bins)
    df['hourCut']=df['hourCut'].astype(str)
    
    bins=pd.IntervalIndex.from_tuples([(-1,2),(2,7),( 7,60),(60,1000)])
    df['tintCut']=pd.cut(df.tint,bins)
    df['tintCut']=df['tintCut'].astype(str)
    
    bins=pd.IntervalIndex.from_tuples([(-1,50),(50,100),(100,1000),(1000,1000000)])
    df['amtCut']=pd.cut(df.amount_usd_in_cents,bins)
    df['amtCut']=df['amtCut'].astype(str)
    
    df['I']=1
    t1=pd.pivot_table(df, values='I', index='merchant', columns='hourCut',aggfunc=np.sum).fillna(0)
    t1.columns=[x+' transaction hour' for x in t1.columns]
    t1=t1.iloc[:,1:]
    t1_p=t1.divide( df_agg.merchant_count, axis=0)
    
    
    df['I']=1
    t2=pd.pivot_table(df, values='I', index='merchant', columns='amtCut',aggfunc=np.sum).fillna(0)
    t2.columns=[x+' $ amount' for x in t2.columns]
    t2=t2.iloc[:,1:]
    t2_p=t2.divide( df_agg.merchant_count, axis=0)
    
    
    df['I']=1
    t3=pd.pivot_table(df, values='I', index='merchant', columns='tintCut',aggfunc=np.sum).fillna(0)
    t3.columns=[x+' day bet trans.' for x in t3.columns]
    t3=t3.iloc[:,1:]
    del t3['nan day bet trans.']
    t3_p=t3.divide( df_agg.merchant_count, axis=0)
    
    
    
    df['I']=1
    t4=pd.pivot_table(df, values='I', index='merchant', columns='dayCut',aggfunc=np.sum).fillna(0)
    t4.columns=[x+' day' for x in t4.columns]
    t4=t4.iloc[:,1:]
    t4_p=t4.divide(df_agg.merchant_count, axis=0)
    
    wk_p=df_agg[['wknd_sum']].divide(df_agg.merchant_count, axis=0)
    df_stat=pd.concat([pd.concat([t1,t2,t3,t4], axis=1), df_agg],axis=1)
    df_stat.columns=[x+'_O' for x in df_stat.columns]
    
    return(pd.concat([df_stat,t1_p,t2_p,t3_p,t4_p, wk_p], axis=1) )
    
time_st=pd.to_datetime('2032-12-31 20:07:57')
time_end=pd.to_datetime('2034-12-31 20:07:57')

df_filtered=generatefeatures(df, time_st, time_end)

df_filtered['logged_amt']=np.log(df_filtered.amount_usd_in_cents_mean_O)    
df_filtered['logged_int']=np.log(df_filtered.tint_mean_O+1)    

featuresCluster=['(18, 23] transaction hour',
       '(8, 18] transaction hour', '(100, 1000] $ amount',
       '(1000, 1000000] $ amount', '(50, 100] $ amount',
       '(2, 7] day bet trans.', '(60, 1000] day bet trans.',
       '(7, 60] day bet trans.', '(15, 31] day',
       'wknd_sum', 'logged_amt', 'logged_int']

featuresClusterO=['(18, 23] transaction hour',
       '(8, 18] transaction hour', '(100, 1000] $ amount',
       '(1000, 1000000] $ amount', '(50, 100] $ amount',
       '(2, 7] day bet trans.', '(60, 1000] day bet trans.',
       '(7, 60] day bet trans.', '(15, 31] day',
       'wknd_sum', 'amount_usd_in_cents_mean_O', 'tint_mean_O']

#### clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
ss=df_filtered.merchant_count_O>2
(scaler.fit(df_filtered[featuresCluster][ss]))
X=scaler.transform(df_filtered[featuresCluster][ss])
from sklearn.cluster import KMeans

wcss=[]
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', 
                                        random_state = 42)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1,11), wcss,'-bo')
plt.title('elbow method')
plt.xlabel('# clusters')
plt.ylabel('wcss')
plt.show()

i=7
km = KMeans(n_clusters = i, init = 'k-means++', 
                                        random_state = 42)
km.fit(X)
#import pickle
#pickle.dump( km, open( "km.pkl", "wb" ) )
#km=pickle.load(open( "km.pkl", "rb" ))

df_filtered['label']=np.nan
df_filtered['label'][ss]=km.labels_
df_filtered[featuresClusterO+['label']].groupby('label').mean().to_csv(r'C:\Users\Min\clusters.csv')
df_filteredO=df_filtered.copy(deep=True)

######## check how many false positive we gets with the following threshold definitions

fixedTh=90
mincap=30
maxcap=180
factorStd=4
mischurn=[]
for i in range(3,10):
    midpt=pd.to_datetime('2033-12-31 20:07:57')+pd.Timedelta(i,unit='M')
    eligiblet = (df.time>pd.to_datetime('2032-12-31 20:07:57')) & (df.time < midpt)
    
    tdf=df[eligiblet]

    tdf['merchant_1_']=tdf.merchant.shift(1)
    tdf['merchant_1']=tdf.merchant.shift(-1)
    tdf['time_1']=tdf.time.shift(-1)
    tdf['tint']= (tdf.time_1-tdf.time).dt.days
    tdf['endt']=tdf.merchant_1 != tdf.merchant     
    tdf['tint'][tdf.endt]= None    
    tdf['tinactive']=np.nan
    tdf['tinactive'][tdf.endt]=(midpt-tdf.time)[tdf.endt].apply(lambda x: x.days)
    tdf_agg=tdf.groupby('merchant').agg({'tint':['mean','std'],'tinactive':'mean'})
    
    tdf_agg.columns=['tintM','tintS','inactiveM']
    churn1MID=tdf_agg.index[(tdf_agg.inactiveM > fixedTh)]
    churn2MID=tdf_agg.index[(tdf_agg.inactiveM > (tdf_agg.tintM + tdf_agg.tintS * factorStd).apply(lambda x: max(min(x,maxcap),mincap) ))]
    
    eligiblet = (df.time> midpt)
    tdf2=df[eligiblet]
    year2MID=tdf2.merchant.unique()
    
    mischurn.append([midpt, sum(churn1MID.isin(year2MID)), sum(churn2MID.isin(year2MID)),len(churn1MID), len(churn2MID), tdf_agg.shape[0]])
    
    
mischurn



fixedTh=90
mincap=30
maxcap=180
factorStd=4
df_filtered.isnull().sum()
import numpy as np
from sklearn.preprocessing import Imputer 
imp = Imputer(missing_values=np.nan, strategy='mean')
imp.fit(df_filtered[[  'tint_mean_O', 'tint_std_O' ]].values)
df_filtered[[  'tint_mean_O', 'tint_std_O' ]]=imp.transform(df_filtered[[ 'tint_mean_O', 'tint_std_O'  ]])

churnETA=(df_filtered.tint_mean_O+df_filtered.tint_std_O * factorStd).apply(lambda x: max(min(int(x),maxcap),mincap) )
#perc1(churnETA)

churnind=df_filtered.tinactive_O>churnETA
churnedID=df_filtered.index[churnind]
churnedTime=df_filtered.endtime_O[churnind] + churnETA.apply(lambda x: np.timedelta64(x,'D'))[churnind]

len(churnedID)/df_filtered.shape[0]
df_filtered['churned']=churnind

##### gather churn information across clusters
tmpp=df_filtered.groupby('label')['churned'].agg(['sum','size'])
tmpp.columns=['Churned merchant', 'Total merchants']
tmpp.index=[int(x+1) for x in tmpp.index.values]
ax=tmpp.plot(kind='bar',title='# of merchant counts and churned per cluster')
ax.set_xlabel('cluster')
ax.set_ylabel('# of merchants')

              
ax=(tmpp.iloc[:,0]/tmpp.iloc[:,1]*100).plot(kind='bar',title='% of churn per cluster')
ax.set_xlabel('cluster')
ax.set_ylabel('% of churn')          


##### train churn model using features from 15 months, 1 quarter ahead prediction


churnobsMonths=3
trainMonths=15
churnstart=pd.to_datetime('2034-12-31 20:07:57')-pd.Timedelta(churnobsMonths,unit='M')
time_st=churnstart-pd.Timedelta(trainMonths,unit='M')
time_end=churnstart

churned2034=churnedID[(churnedTime>churnstart)  ]
churned2033=churnedID[churnedTime<churnstart]

df_train_2033=generatefeatures(df, time_st, time_end)
del df_train_2033['endtime_O']
df_train_2033.isnull().sum()

df_train_2033 = df_train_2033[~(df_train_2033.index.isin(churned2033))]
df_train_2033.isnull().sum()

imp = Imputer(missing_values=np.nan, strategy='mean')
imp.fit(df_train_2033.values)
df_train_2033[:]=imp.transform(df_train_2033)


import seaborn as sns
corr=df_train_2033.corr()
kot=corr[corr>0.95]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap='Greens')


y=df_train_2033.index.isin(churned2034)
xTrain, xTest, yTrain, yTest = train_test_split(df_train_2033, y, test_size=0.25, random_state=5)

model_params = {
    'n_estimators': [200, 400], 
    'max_depth': [5, 20, 50]
}
rf_model = RandomForestClassifier(random_state=1) 
grid_clf = GridSearchCV(rf_model, model_params, cv=5)
RF = grid_clf.fit(xTrain, yTrain)
sc_RF=RF.score(xTest, yTest)
RF.best_params_
RF_p=RF.predict_proba(xTest)

precision, recall, thresholds=precision_recall_curve(yTest, RF_p[:,1])
auc0=auc(recall, precision)
false_positive_rate, true_positive_rate, threshold=roc_curve(yTest, RF_p[:,1])
roc_auc=auc(false_positive_rate, true_positive_rate)


plt.plot(false_positive_rate, true_positive_rate, color='b',  label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy',   linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.legend(loc="lower right")
plt.show()



plt.plot(recall,precision, color='b',  label='ROC curve (area = %0.2f)' % auc0)
plt.plot([0, 1], [0, 0], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision') 
plt.legend(loc="lower right")
plt.show()



coef=[]
cols=df_train_2033.columns
for col in cols:
    coef=coef+[np.corrcoef(df_train_2033[col],y)[0,1]]
corcoef=pd.DataFrame({'corr_coef':coef, 'colname':cols})
corcoef['abs_corr_coef']=abs(corcoef.corr_coef)
corcoef=corcoef.sort_values('abs_corr_coef')


ftrImp=pd.DataFrame({'imp_RF':RF.best_estimator_.feature_importances_, 'colname':cols})
ftrImp=ftrImp.merge(corcoef, on='colname',how='left')
ftrImp=ftrImp.sort_values(['imp_RF'], ascending=[0]) 

f, ax = plt.subplots(figsize=(6, 15)) 
sns.set_color_codes("pastel")
sns.barplot(y="colname", x="imp_RF", data=ftrImp,
            label="randomForest", color="b")

sns.set_color_codes("muted")
sns.barplot(y="colname", x="corr_coef", data=ftrImp,
            label="PearsonCorrelation", color="g",alpha=0.3)

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="feature importance")
sns.despine(left=True, bottom=True)

n=20

RF1 = RandomForestClassifier( n_estimators= 400, max_depth=20, class_weight='balanced').fit(xTrain[ftrImp.colname[:n]],yTrain)
sc_RF1=RF1.score(xTest[ftrImp.colname[:n]], yTest) 
RF1_p=RF1.predict_proba(xTest[ftrImp.colname[:n]])

precision, recall, thresholds=precision_recall_curve(yTest, RF1_p[:,1])
auc1=auc(recall, precision)
false_positive_rate, true_positive_rate, threshold=roc_curve(yTest, RF1_p[:,1])
roc_auc1 = auc(false_positive_rate, true_positive_rate)


RF2 = RandomForestClassifier( n_estimators= 400, max_depth=20, class_weight='balanced').fit(xTrain.tinactive_O.values.reshape(-1,1),yTrain)
sc_RF2=RF2.score(xTest.tinactive_O.values.reshape(-1,1), yTest)

print([churnobsMonths, trainMonths, sc_RF, sc_RF1, sc_RF2, auc0, auc1, roc_auc, roc_auc1])


#grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]} 
#logreg=LogisticRegression()
#logreg_cv=GridSearchCV(logreg,grid,cv=10)
#logreg_cv.fit(xTrain,yTrain)
#logreg_cv.score(xTest,yTest)
#




import scikitplot as skplt
skplt.metrics.plot_lift_curve(y_true=yTest, y_probas = RF_p)

df_score=generatefeatures(df, pd.to_datetime('2034-12-31 20:07:57')-pd.Timedelta(trainMonths,unit='M'), pd.to_datetime( '2034-12-31 20:07:57') )
del df_score['endtime_O']
df_score=df_score[~(df_score.index.isin(churnedID))]

imp = Imputer(missing_values=np.nan, strategy='mean')
imp.fit(df_score.values)
df_score[:]=imp.transform(df_score)

p=RF.predict_proba(df_score)

tmp=RF_p[:,1]
tmp.sort()
thresholdP=np.percentile(tmp, 70)


df_score['churnLabel']='Not churned'
df_score['churnLabel'][p[:,1]>=thresholdP]='Likely to churn' 
df_score.groupby('churnLabel').size().plot(kind='bar')
sum(df_score['churnLabel']=='Likely to churn' )/df_score.shape[0]


df_score['propensityScore']=p[:,1]
df_score[['propensityScore','churnLabel']].to_csv(r'C:\Users\Min\scored.csv')


 
df_filtered['churned'].to_csv(r'C:\Users\Min\churned.csv')

df_filtered_O['churned'] =  df_filtered_O.tinactive>90
df_filtered_O['churned'] =  df_filtered['churned']
perc2(df_filtered_O.tint_mean_O.dropna())

sns.distplot(df_filteredO.tint_mean_O[df_filteredO.churned].dropna(), color="dodgerblue", label="churned") 
sns.distplot( df_filteredO.tint_mean_O[~df_filteredO.churned].dropna(), color="orange", label="not churned") 
plt.legend(loc='upper right') 
plt.xlabel('# days between first and latest transactions') 
plt.title('Density of inactive days comparison (churned vs. active users)') 
plt.show()
df_filtered['loggedTintMean']= np.log(df_filtered.tint_mean_O+1)
fig =  sns.boxplot(x="churned", y="loggedTintMean", data=df_filtered, palette="Set3")
fig.set(  ylabel='logged average # days between transaction')

        
sns.distplot(df_filteredO.tint_std_O[df_filtered.churned].dropna(), color="dodgerblue", label="churned") 
sns.distplot( df_filteredO.tint_std_O[~df_filtered.churned].dropna(), color="orange", label="not churned") 
plt.legend(loc='upper right') 
plt.xlabel('# days between first and latest transactions') 
plt.title('Density of identified account tenure comparison (churned vs. active users)') 
plt.show()
df_filtered['loggedTintStd']= np.log(df_filtered.tint_mean_O+1)
fig =  sns.boxplot(x="churned", y="loggedTintStd", data=df_filtered, palette="Set3")
fig.set(  ylabel='# days holding the account')

 
fig =  sns.boxplot(x="churned", y="length_O", data=df_filtered, palette="Set3")
fig.set(  ylabel='# days holding the account')
        
fig =  sns.boxplot(x="churned", y="tinactive_O", data=df_filtered, palette="Set3")
fig.set(  ylabel='# days from the latest transaction')
        
# check if the performance of the model stays similar 3 month in the past
churnobsMonths=3
trainMonths=15
churnstart=pd.to_datetime('2034-09-30 20:07:57')-pd.Timedelta(churnobsMonths,unit='M')
time_st=churnstart-pd.Timedelta(trainMonths,unit='M')
time_end=churnstart

churned2034=churnedID[(churnedTime>churnstart)  ]
churned2033=churnedID[churnedTime<churnstart]

df_train_2033=generatefeatures(df, time_st, time_end)
del df_train_2033['endtime_O']
df_train_2033.isnull().sum()

df_train_2033 = df_train_2033[~(df_train_2033.index.isin(churned2033))]
df_train_2033.isnull().sum()

imp = Imputer(missing_values=np.nan, strategy='mean')
imp.fit(df_train_2033.values)
df_train_2033[:]=imp.transform(df_train_2033)


y=df_train_2033.index.isin(churned2034) 
sc_RF=RF.score(df_train_2033, y) 
p1=RF.predict_proba(df_train_2033)

precision, recall, thresholds=precision_recall_curve(yTest, p1[:,1])
auc0=auc(recall, precision)
false_positive_rate, true_positive_rate, threshold=roc_curve(yTest, p1[:,1])
roc_auc=auc(false_positive_rate, true_positive_rate)



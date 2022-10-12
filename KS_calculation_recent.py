# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:13:08 2021

@author: korni
"""
#%%
import os
from scipy.stats import ks_2samp
import sys
import pandas
from scipy.stats import zscore
import numpy as np

pathName = os.getcwd()
#read the list of files csv in the path
numFiles = []
fileNames = os.listdir(pathName)
for fileNames in fileNames:
    if fileNames.endswith(".csv"):
       numFiles.append(fileNames)
        #runs through the Bands and calculates KS
for i in numFiles:
    file = open(os.path.join(pathName, i))
    reader = pandas.read_csv(file, index_col=None)
    #removing outliers
    a_raw = reader.iloc[:,0].dropna()
    z_score_a = zscore(a_raw)
    #gets the scores for both +_ and keeps only the fltered ones
    abs_zscores = np.abs(z_score_a)
    #checks who are passing the criteria
    filtered_a = (abs_zscores < 3)
    a=a_raw[filtered_a]
    #the same for b
    b_raw = reader.iloc[:,1].dropna()
    z_score_b = zscore(b_raw)
    abs_z_scores = np.abs(z_score_b)
    filtered_b = (abs_z_scores < 3)
    b=b_raw[filtered_b]
    ks_statistic, p_value = ks_2samp(a,b,mode='exact')
    K= ks_statistic,p_value
    #for the FR columns
    c_raw = reader.iloc[:,2].dropna()
    z_score_c = zscore(c_raw)
    #gets the scores for both +_ and keeps only the fltered ones
    abs_scores = np.abs(z_score_c)
    #checks who are passing the criteria
    filtered_c = (abs_scores < 3)
    c=c_raw[filtered_c]
    d_raw = reader.iloc[:,3].dropna()
    z_score_d = zscore(d_raw)
    #gets the scores for both +_ and keeps only the fltered ones
    abs_Zscores = np.abs(z_score_d)
    #checks who are passing the criteria
    filtered_d = (abs_scores < 3)
    d=d_raw[filtered_d]
    ks_statistic, p_value = ks_2samp(c,d,mode='exact')
    S= ks_statistic,p_value
    sys.stdout = open('FP-June23_KS_2016.txt', 'a') 
    print ('filename: ',file, 'ks column 0 and 1: ', K, 'ks column 2 and 3: ', S)     

#%%
from scipy.stats import zscore
import numpy as np
import matplotlib as plt
 
url = "C:/Users/korni/Desktop/GalaktobourekO/FP/June23/B01June23.csv"

#read bands for KS
B1  = pandas.read_csv(url, index_col=None, names=None)
# B1= B1.dropna()
# C2=B1
# #print(10*(np.log10(C2)))
# #removing outliers
# z_scores = zscore(B1)
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# C2_=B1
# B1 = B1[filtered_entries]
# B1.hist(cumulative=True, density=1, bins=100)

#KS separability
#FYI_MYI_steep
for k in B1.keys():
    a = B1[B1.columns[0]]
    b = B1[B1.columns[1]]
   
#from scipy.stats import epps_singleton_2samp  
#Epps, pval= epps_singleton_2samp (a,b)      
from scipy.stats import ks_2samp
ks_statistic, p_value = ks_2samp(a,b,mode='exact')
K= ks_statistic,p_value
#KS separability - FYI_MYI shallo
for l in B1 .keys():
    c = B1[B1.columns[2]]
    d = B1[B1.columns[3]]
from scipy.stats import ks_2samp
ks_statistic, p_value = ks_2samp(c,d, mode= 'exact')

S= ks_statistic,p_value

# from scipy.stats import ks_2samp
# ks_statistic, p_value = ks_2samp(e,f,mode='exact')
# K= ks_statistic,p_value
#KS separability - FYI_MYI shallo
for l in B1 .keys():
    e = B1[B1.columns[4]]
    f = B1[B1.columns[5]]
from scipy.stats import ks_2samp
ks_statistic, p_value = ks_2samp(e,f, mode= 'exact')

T= ks_statistic,p_value

print (K,S,T)
#   #FYI_MYI_steep
# for k in B1.keys():
#     a_ = B1[B1.columns[0]]
#     b_ = B1[B1.columns[3]]
# #from scipy.stats import epps_singleton_2samp  
# #Epps, pval= epps_singleton_2samp (a,b)      
# from scipy.stats import ks_2samp
# ks_statistic, p_value = ks_2samp(a_,b_,mode='exact')
# L= ks_statistic,p_value
#   #FYI_MYI_steep
# for k in B1.keys():
#     c_ = B1[B1.columns[1]]
#     d_ = B1[B1.columns[2]]
# #from scipy.stats import epps_singleton_2samp  
# #Epps, pval= epps_singleton_2samp (a,b)      
# from scipy.stats import ks_2samp
# ks_statistic, p_value = ks_2samp(c_,d_,mode='exact')
# M= ks_statistic,p_value


#%%
# from scipy.stats import wasserstein_distance
# wd = wasserstein_distance(a,b)
# from scipy.stats import energy_distance
# ed = energy_distance(a, b)
#from scipy.stats import epps_singleton_2samp  
#Epps, pval= epps_singleton_2samp (a,b) 
#%%
import matplotlib.pyplot as plt
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "C:/Users/korni/Desktop/GalaktobourekO/CP_KS_heatmat_2016_final.csv"
data = pandas.read_csv(url)
plt.figure()
y_axis_labels= ['S0', 'S1', 'S2', 'RRRL', 'Alpha_s','mx_e','mx_vol','mx_odd']
x_axis_labels= ['LW_NR','LW_FR','MO_NR','MO_FR','PO_NR','PO_FR','PE_NR','PE_FR','L-PE_NR', 'L-PE_FR','PD_NR','PD_FR','L-PD_NR','L-PD_FR']
#x_axis_labels= ['LW_NR','LW_FR','PE_NR','PE_FR','PD_NR','PD_FR']
#'HH', 'HV','VH','VV']
#plt.title('KS 2018', fontsize = 25, color= "c")
fig= sns.heatmap(data.drop(['2016'],axis=1), annot=True, annot_kws={"size":20} , yticklabels = y_axis_labels, cmap="coolwarm")
#plt.xticks(range(len(data.columns-1)), data.columns, fontsize=29)
cbar=fig.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
fig.set_yticklabels(fig.get_ymajorticklabels(), fontsize = 20)
fig.set_xticklabels(fig.get_xmajorticklabels(), fontsize =19)
fig.xaxis.set_ticks_position('top')
plt.xticks(range(len(data.columns-1)), data.columns, fontsize=20)
fig.savefig('updatedKS2018_CP-Jan.png',dpi=300,facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False)


#%%
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import numpy as np

url = "C:/Users/korni/Desktop/GalaktobourekO/FP_KS_heatmap.csv"
data = pandas.read_csv(url)
plt.figure()
y_axis_labels= ['HH', 'HV', 'VH', 'VV', 'VV/HH', 'HV/HH', 'VV/VH', 'HVxHH', 'VHxVV', 'HHxVV']
x_axis_labels= ['LW_NR','LW_FR','MO_NR','MO_FR','PO_NR','PO_FR','PE_NR','PE_FR','L-PE_NR','L-PE_FR','PD_NR','PD_FR']
#'HH', 'HV','VH','VV']
#plt.title('KS 2018', fontsize = 25, color= "c")
fig= sns.heatmap(data.drop(['2018'],axis=1), annot=True, annot_kws={"size":20} , yticklabels = y_axis_labels, cmap="coolwarm")
#plt.xticks(range(len(data.columns-1)), data.columns, fontsize=29)

cbar=fig.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
fig.set_yticklabels(fig.get_ymajorticklabels(), fontsize = 20)
fig.set_xticklabels(fig.get_xmajorticklabels(), fontsize =19)
fig.xaxis.set_ticks_position('top')
plt.xticks(range(len(data.columns-1)), data.columns, fontsize=20)
fig.savefig('KSheatmap_CP.png',dpi=300,facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False)


#%%
import seaborn  as sns
import matplotlib.pyplot as plt
url = "C:/Users/korni/Desktop/GalaktobourekO/CP_KS_heatmat_2016_final.csv"
data2016 = pandas.read_csv(url)

url = "C:/Users/korni/Desktop/GalaktobourekO/CP_KS_heatmat_2017_final.csv"
data2017 = pandas.read_csv(url)

url = "C:/Users/korni/Desktop/GalaktobourekO/CP_KS_heatmat_2018_final.csv"
data2018 = pandas.read_csv(url)
y_axis_labels= ['S0', 'S1', 'S2', 'S3', 'DoP', 'DoD', 'CPR', 'LPR', 'RP', 'Alpha_s','Conf','mchi_e','mchi_v','mchi_o']
s= ['LW_NR','LW_FR','MO_NR','MO_FR','PO_NR','PO_FR','PE_NR','PE_FR','PD_NR','PD_FR']

#%%
import seaborn  as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas
import numpy as np

url = "C:/Users/korni/Desktop/GalaktobourekO/FP_KS_heatmat_2016_last_update.csv"
data2016 = pandas.read_csv(url,index_col=0)

url = "C:/Users/korni/Desktop/GalaktobourekO/FP_KS_heatmat_2017_last_update.csv"
data2017 = pandas.read_csv(url,index_col=0)

url = "C:/Users/korni/Desktop/GalaktobourekO/FP_KS_heatmat_2018_last_update.csv"
data2018 = pandas.read_csv(url,index_col=0)
y_axis_labels= ['HH', 'HV', 'VH', 'VV', 'VV/HH', 'HV/HH', 'HV*HH', 'Pauli-even', 'Pauli-vol', 'Pauli-odd']
s= ['LW_NR','LW_FR','MO_NR','MO_FR','PO_NR','PO_FR','PE_NR','PE_FR','PD_NR','PD_FR']

# y_axis_labels= ['S0', 'S1', 'S2', 'RRRL', 'Alpha_s','mx_e','mx_vol','mx_odd']
# #s= ['LW_NR','LW_FR','MO_NR','MO_FR','PO_NR','PO_FR','PE_NR','PE_FR','L-PE_NR', 'L-PE_FR','PD_NR','PD_FR','L-PD_NR','L-PD_FR']

# s= ['LW_NR','LW_FR','MO_NR','MO_FR','PO_NR','PO_FR','PE_NR','PE_FR', 'PD_NR','PD_FR']
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
       # Sample figsize in inches
plt.close('all')
 
f,(ax1,ax2,ax3, axcb) = plt.subplots(1,4, 
         gridspec_kw={'width_ratios':[1.1,1.6,2,0.08]})
ax1.get_shared_y_axes().join(ax2,ax3)


g1 = sns.heatmap(data2016, annot=True, yticklabels = y_axis_labels, cmap="coolwarm",cbar=False,ax=ax1)
g1.set_ylabel('')
#g1.xaxis.set_ticks_position('top')
g1.set_xlabel('')
plt.setp(g1.get_xticklabels(), rotation=45, size=12)
g2 = sns.heatmap(data2017, annot=True, yticklabels = y_axis_labels, cmap="coolwarm",cbar=False,ax=ax2)
g2.set_ylabel('')
g2.set_xlabel('')
g2.set_yticks([])
#g2.xaxis.set_ticks_position('top')
plt.setp(g2.get_xticklabels(), rotation=45, size=12)
g3 = sns.heatmap(data2018, annot=True, yticklabels = y_axis_labels,xticklabels= s,cbar_kws={'label': 'Kolmogorov-Smirnov separability test'}, cmap="coolwarm",cbar_ax=axcb, ax=ax3)
axcb.yaxis.label.set_size(14)
g3.set_ylabel('')
g3.set_xlabel('')
g3.set_yticks([])
plt.setp(g3.get_xticklabels(), rotation=45, size=12)
#g3.xaxis.set_ticks_position('top')
#plt.setp(g3.get_xticklabels(), rotation=45)
#plt.rcParams["font.weight"] = "bold"
g1.set_title('2016', fontweight='bold', size=14)
g2.set_title('2017', fontweight='bold', size=14)
g3.set_title('2018', fontweight='bold', size=14)
f.suptitle("C-band RS2 - Fully polarimetric (FP)", size=15)

# plt.xticks(range(len(data.columns-1)), data.columns, fontsize=15, pos= 'top',rotation=45)
# g1.tick_params(labelsize=15)
# g2.tick_params(labelsize=15)
# g3.tick_params(labelsize=15)

f.savefig('KS-heatmap_FP.png',dpi=300,facecolor='w', edgecolor='w',orientation='portrait', format='png')

#%%
import seaborn  as sns

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas
import numpy as np

plt.close('all')
url = "C:/Users/korni/Desktop/GalaktobourekO/CP_KS_heatmat_2016_last_update.csv"
data2016 = pandas.read_csv(url,index_col=0)

url = "C:/Users/korni/Desktop/GalaktobourekO/CP_KS_heatmat_2017_last_update.csv"
data2017 = pandas.read_csv(url,index_col=0)

url = "C:/Users/korni/Desktop/GalaktobourekO/CP_KS_heatmat_2018_last_update.csv"
data2018 = pandas.read_csv(url,index_col=0)
# y_axis_labels= ['HH', 'HV', 'VH', 'VV', 'VV/HH', 'HH/HV', 'HV*HH', 'HH*VV','HH+VV','HH-VV', 'Pauli-even', 'Pauli-vol', 'Pauli-odd']
# s= ['LW_NR','LW_FR','MO_NR','MO_FR','PO_NR','PO_FR','PE_NR','PE_FR','PD_NR','PD_FR']


y_axis_labels= ['S0', 'S2', 'RR/RL','mχ_e','mχ_vol','mχ_odd']
#s= ['LW_NR','LW_FR','MO_NR','MO_FR','PO_NR','PO_FR','PE_NR','PE_FR','L-PE_NR', 'L-PE_FR','PD_NR','PD_FR','L-PD_NR','L-PD_FR']

s= ['LW_NR','LW_FR','MO_NR','MO_FR','PO_NR','PO_FR','PE_NR','PE_FR', 'PD_NR','PD_FR']


 
f,(ax1,ax2,ax3, axcb) = plt.subplots(1,4, 
         gridspec_kw={'width_ratios':[1.1,1.6,2,0.08]})
ax1.get_shared_y_axes().join(ax2,ax3)


g1 = sns.heatmap(data2016, annot=True, yticklabels = y_axis_labels, cmap="coolwarm",cbar=False,ax=ax1)
g1.set_ylabel('')
#g1.xaxis.set_ticks_position('top')
g1.set_xlabel('')
plt.setp(g1.get_xticklabels(), rotation=45, size=12)
g2 = sns.heatmap(data2017, annot=True, yticklabels = y_axis_labels, cmap="coolwarm",cbar=False,ax=ax2)
g2.set_ylabel('')
g2.set_xlabel('')
g2.set_yticks([])
#g2.xaxis.set_ticks_position('top')
plt.setp(g2.get_xticklabels(), rotation=45, size=12)
g3 = sns.heatmap(data2018, annot=True, yticklabels = y_axis_labels,xticklabels= s,cbar_kws={'label': 'Kolmogorov-Smirnov separability test'}, cmap="coolwarm",cbar_ax=axcb, ax=ax3)
axcb.yaxis.label.set_size(14)
g3.set_ylabel('')
g3.set_xlabel('')
g3.set_yticks([])
plt.setp(g3.get_xticklabels(), rotation=45, size=12)
#g3.xaxis.set_ticks_position('top')
#plt.setp(g3.get_xticklabels(), rotation=45)
#plt.rcParams["font.weight"] = "bold"
g1.set_title('2016', fontweight='bold', size=14)
g2.set_title('2017', fontweight='bold', size=14)
g3.set_title('2018', fontweight='bold', size=14)
f.suptitle("C-band RS2 - Simulated Compact Polarimetric (CP) ", size=15)

# plt.xticks(range(len(data.columns-1)), data.columns, fontsize=15, pos= 'top',rotation=45)
# g1.tick_params(labelsize=15)
# g2.tick_params(labelsize=15)
# g3.tick_params(labelsize=15)
#%%

f,(ax1,ax2,ax3, axcb) = plt.subplots(1,4, 
         gridspec_kw={'width_ratios':[0.4,0.7,1,0.08]})
ax1.get_shared_y_axes().join(ax2,ax3)
#figsize=(15,9)

g1 = sns.heatmap(data2016, annot=True, yticklabels = y_axis_labels, cmap="coolwarm",cbar=False,ax=ax1)
g1.set_ylabel('')
g1.xaxis.set_ticks_position('top')

#g1.set_xlabel('')
g2 = sns.heatmap(data2017, annot=True, yticklabels = y_axis_labels, cmap="coolwarm",cbar=False,ax=ax2)
g2.set_ylabel('')
#g2.set_xlabel('')
g2.set_yticks([])
g2.xaxis.set_ticks_position('top')

g3 = sns.heatmap(data2018, annot=True, yticklabels = y_axis_labels, cmap="coolwarm",cbar_ax=axcb, ax=ax3)
g3.set_ylabel('')
#g3.set_xlabel('')
g3.set_yticks([])
g3.xaxis.set_ticks_position('top')

#plt.xticks(range(len(data.columns-1)), data.columns, fontsize=15, pos= 'top')
plt.tight_layout()

g1.tick_params(labelsize=13)
g2.tick_params(labelsize=13)
g3.tick_params(labelsize=13)
plt.setp(f.get_xticklabels(), rotation=45)
f.savefig('RESULT.png',dpi=300,facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False)


#, bbox_inches="tight"

#%%


#%%
import pandas
import seaborn  as sns
import matplotlib.pyplot as plt
url = "C:/Users/korni/Desktop/GalaktobourekO/ALOS_heatmap_2016_final.csv"
data2016 = pandas.read_csv(url,index_col=0)

url = "C:/Users/korni/Desktop/GalaktobourekO/ALOS_heatmap_2018_final.csv"
data2018 = pandas.read_csv(url,index_col=0)
y_axis_labels= ['HH', 'HV', 'VH', 'VV', 'VV/HH', 'HH/HV', 'HV*HH','Pauli-even', 'Pauli-vol', 'Pauli-odd']
#,'HH+VV','HH-VV', 'Pauli-even', 'Pauli-vol', 'Pauli-odd'
s= ['LW_NR','LW_FR','MO_NR','MO_FR','PO_NR','PO_FR','PE_NR','PE_FR','PD_NR','PD_FR']

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

plt.close('all')
f,(ax1,ax2,axcb) = plt.subplots(1,3, 
         gridspec_kw={'width_ratios':[0.1,0.5,0.08]})
ax1.get_shared_y_axes().join(ax2)

g1 = sns.heatmap(data2016, annot=True, yticklabels = y_axis_labels, cmap="coolwarm",cbar=False,annot_kws={"fontsize":11}, ax=ax1)
g1.set_ylabel('')
#g1.xaxis.set_ticks_position('top')
plt.setp(g1.get_xticklabels())
g1.set_xlabel('')
g2 = sns.heatmap(data2018, annot=True, yticklabels = y_axis_labels,cmap="coolwarm",annot_kws={"fontsize":11},cbar_kws={'label': 'Kolmogorov-Smirnov separability test'},cbar_ax=axcb,ax=ax2)
g2.set_ylabel('')
g2.set_xlabel('')
plt.setp(g2.get_xticklabels())
g2.set_yticks([])
axcb.yaxis.label.set_size(15)
#g2.xaxis.set_ticks_position('top')
#plt.setp(g2.get_xticklabels())


#plt.rcParams["font.weight"] = "bold"
g1.set_title('2016', fontweight='bold')
g2.set_title('2018', fontweight='bold')

f.suptitle("    L-band PS2", size=14);

#plt.xticks(range(len(data.columns-1)), data.columns, fontsize=25, pos= 'top',rotation=45)
g1.tick_params(labelsize=14)
g2.tick_params(labelsize=14)
axcb.tick_params(labelsize=14)
plt.xticks(rotation = 45)

f.savefig('KS-heatmap_Lband.png',dpi=400,facecolor='w', edgecolor='w',orientation='portrait', format='png')

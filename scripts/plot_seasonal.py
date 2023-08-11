#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%% Import

from netCDF4 import Dataset
import numpy as np
import time
import xarray as xr
import glob
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.ticker as ticker
import pymannkendall as mk
import matplotlib.patches as mpatches


# In[2]:


#%% Read in files

# Marker to seperate annual data by grid configuration, or not
ann_sep = True
sea_sep = True

maindir = "scratch/TempestExtremes/"

f19 = "cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs"
f09 = "cam6_2_022.se_FHIST_f09_f09_mg17_1800pes_200507_mg3-Nx5yrs"
pg2 = "cam6_2_022.se_FHIST_ne30pg2_ne30pg2_mg17_1800pes_200507_mg3-Nx5yrs"
pg3 = "cam6_2_022.se_FHIST_ne30pg3_ne30pg3_mg17_1800pes_200507_mg3-Nx5yrs"
arc = "cam6_2_022.se_FHIST_ne0np4.ARCTIC.ne30x4_mt12_7680pes_200507_mg3-Nx2yrs"
arcG = "cam6_2_022.se_FHIST_ne0np4.ARCTICGRIS.ne30x8_mt12_7680pes_200510_mg3-Nx1yrs"
era5 = "ERA5"
merra2 = "MERRA2"

E_dir = "ESMF/"
TR_dir = "TempestRemap/"

fnames = [f19,f09,pg2,pg3,arc,arcG,era5, merra2]
labels = ['f19','f09','ne30pg2','ne30pg3','ARCTIC','ARCTICGRIS','ERA5', 'MERRA2']
colors = ['blue','deepskyblue','green','lime','purple','magenta','black', 'grey']

# Setting up "days since" to datetime
day_amount = 7300.5                                        
start_datetime = datetime(1979,1,1,0,0,0)

# ERA5 date information
hour_amount_ERA5 = 701250
start_datetime_ERA5 = datetime(1900,1,1,0,0,0)

hours_amount_MERRA2 = 4017.75
start_datetime_MERRA2 = datetime(1970,1,1,0,0,0)

#GrIS ID files
workdir = "work/AR_GrIS/"
workdir_scrip = "../data/seasonal/"

# Data containing blob ID info
ids = []
for case in fnames:
    name = glob.glob(workdir_scrip+E_dir+"f19/"+case+"/*grisids.nc")
    file = Dataset(name[0])
    ids.append(file)
    
    name = glob.glob(workdir_scrip+E_dir+"ne30pg2/"+case+"/*grisids.nc")
    file = Dataset(name[0])
    ids.append(file)
    
    name = glob.glob(workdir_scrip+TR_dir+"f19/"+case+"/*grisids.nc")
    file = Dataset(name[0])
    ids.append(file)
    
    name = glob.glob(workdir_scrip+TR_dir+"ne30pg2/"+case+"/*grisids.nc")
    file = Dataset(name[0])
    ids.append(file)

# Data containing time info
tcenter = []
for case in fnames:
    name = glob.glob(workdir_scrip+E_dir+"f19/"+case+"/*tcenters.nc")
    file = Dataset(name[0])
    tcenter.append(file)
    
    name = glob.glob(workdir_scrip+E_dir+"ne30pg2/"+case+"/*tcenters.nc")
    file = Dataset(name[0])
    tcenter.append(file)

    name = glob.glob(workdir_scrip+TR_dir+"f19/"+case+"/*tcenters.nc")
    file = Dataset(name[0])
    tcenter.append(file)
    
    name = glob.glob(workdir_scrip+TR_dir+"ne30pg2/"+case+"/*tcenters.nc")
    file = Dataset(name[0])
    tcenter.append(file) 


# In[3]:


# Define function to populate annual dataframes

def annual_df(df, df_annual):
    df_annual['year'] = df.index.copy()
    df_annual.set_index('year',inplace = True)
    df_annual['total'].fillna(0,inplace=True)

    df_annual = df_annual.join(df['count'])
    df_annual['total'] = df_annual['total'] + df_annual['count']
    df_annual.drop(columns={'count'}, inplace = True)
    
    return df_annual

def seas_df(df, df_seasonal):
    df_temp = df.copy()
    df_temp['season'] = df.index.month%12//3+1
    df_seasonal = df_temp.groupby(['season'])['binary'].sum()
    
    return df_seasonal

def yearly_seasonal(work_year,df_test, df_seasonal, counter):
    while work_year < 1999:
        df_temp = df_test.copy()
        for index, row in df_temp.iterrows():
            if index.year != year:
                try:
                    df_temp.drop(index, axis = 0, inplace = True)
                except:
                    pass
            else:
                pass
        df_temp['season'] = df_temp.index.month%12//3+1
        df_temp = df_temp.groupby(['season'])['binary'].sum()
        df_seasonal.iloc[0:4,counter] = df_temp[0:4]
        counter = counter + 1
        work_year = work_year + 1
        
    return counter
    return df_seasonal


# In[4]:


# Lookup datetime of each grisid based on var

names_list = []
years = np.arange(1979,1999,1)
remaps = ["E-f19-","E-pg2-","TR-f19-","TR-pg2-"]

years_merra2 = np.arange(1980,2000,1)

index_list = [1,2,3,4]

# Creating list that will hold each of names
for name in remaps:
    for i in years:
        add = name+str(i)
        names_list.append(add)
        
# dataframes to hold annual data
df_ann = pd.DataFrame(columns=['year','total'])
df_ann_f19 = pd.DataFrame(columns=['year','total'])
df_ann_f09 =  pd.DataFrame(columns=['year','total'])
df_ann_pg2 = pd.DataFrame(columns=['year','total'])
df_ann_pg3 = pd.DataFrame(columns=['year','total'])
df_ann_A = pd.DataFrame(columns=['year','total'])
df_ann_AG = pd.DataFrame(columns=['year','total'])
df_ann_era5 = pd.DataFrame(columns=['year','total'])
df_ann_merra2 = pd.DataFrame(columns=['year','total'])

# dataframes to hold seasonal info
df_sea_f19 = pd.DataFrame(index=index_list,columns=names_list)
df_sea_f09 =  pd.DataFrame(index=index_list,columns=names_list)
df_sea_pg2 = pd.DataFrame(index=index_list,columns=names_list)
df_sea_pg3 = pd.DataFrame(index=index_list,columns=names_list)
df_sea_A = pd.DataFrame(index=index_list,columns=names_list)
df_sea_AG = pd.DataFrame(index=index_list,columns=names_list)
df_sea_era5 = pd.DataFrame(index=index_list,columns=names_list)

del(names_list)
names_list = []
# Creating list that will hold each of names
for name in remaps:
    for i in years_merra2:
        add = name+str(i)
        names_list.append(add)

df_sea_merra2 = pd.DataFrame(index=index_list,columns=names_list)


# In[15]:


i = 0

while i < 32: 
    print(i)
    # This is good for accessing keys
    var = ids[i]['grisid'][:].tolist()
    time = tcenter[i].variables['time_center'][:].tolist()
    data=[var,time]
    #print(data)
    
    df= pd.DataFrame(data)
    df=df.T #Transpose data to switch x/y
    df.columns=["AR_ID","days_since"]
    df['datetime'] = 0
    
    #print(df['days_since'])
    
    if i < 24:
        for index, row in df.iterrows():
            df.loc[index,'datetime'] = start_datetime + timedelta(days = df.loc[index,'days_since'])
    elif i > 23 and i < 28:
        for index, row in df.iterrows():
            if np.isnan(df.loc[index,'days_since']):
                pass
            else:
                df.loc[index,'datetime'] = start_datetime_ERA5 + timedelta(hours = df.loc[index,'days_since']) 
        df = df.dropna()
    else:
        for index, row in df.iterrows():
            df.loc[index,'datetime'] = start_datetime_MERRA2 + timedelta(days = df.loc[index,'days_since'])  
    df['binary'] = 1
    
    df.drop(columns=['days_since','AR_ID'], inplace = True)
    
    # Annual sum of ARs
    df.set_index('datetime', inplace = True)
    
    df = df.resample("1Y").sum()
    #print(df)
    df['year'] = df.index.year
    df.set_index('year', inplace = True)
    
    df2 = pd.DataFrame([df.index, df['binary']])
    df2 = df2.T
    df2.columns=['year','count']
    df2.set_index('year', inplace = True)
    
    df_ann = annual_df(df2,df_ann)
    
    if ann_sep == True:
        
        if i < 4:
            df_ann_f19 = annual_df(df2,df_ann_f19)
        elif i > 3 and i < 8:
            df_ann_f09 = annual_df(df2,df_ann_f09)
        elif i > 7 and i < 12:
            df_ann_pg2 = annual_df(df2,df_ann_pg2)
        elif i > 11 and i < 16:
            df_ann_pg3 = annual_df(df2,df_ann_pg3)
        elif i > 15 and i < 20:
            df_ann_A = annual_df(df2,df_ann_A)
        elif i > 19 and i < 24:
            df_ann_AG = annual_df(df2,df_ann_AG)
        elif i > 23 and i < 28:
            df_ann_era5 = annual_df(df2,df_ann_era5)
        else:
            df_ann_merra2 = annual_df(df2,df_ann_merra2)
    
    i = i + 1


# In[16]:


# Divide all dfs by number of members included
ann_dfs = [df_ann_f19, df_ann_f09, df_ann_pg2,df_ann_pg3,df_ann_A,df_ann_AG,df_ann_era5, df_ann_merra2]

for df in ann_dfs:
    df['total'] = df['total']/4

#for df in sea_dfs:
#    df = df/4
    #mk.originaltest(df['total'])

df_ann['total'] = df_ann['total']/24


# In[23]:


# Code to create dataframes of yearly ARs

i = 0
count_f19 = 0
count_f09 = 0
count_pg2 = 0
count_pg3 = 0
count_A = 0
count_AG = 0
count_era5 = 0
count_merra2 =0

while i < 32:
    #print(i)
    # This is good for accessing keys
    var = ids[i]['grisid'][:].tolist()

    time = tcenter[i].variables['time_center'][:].tolist()

    data=[var,time]
    
    df= pd.DataFrame(data)
    df=df.T #Transpose data to switch x/y
    df.columns=["AR_ID","days_since"]
    df['datetime'] = 0
    
    #print(df['days_since'])
    
    if i < 24:
        for index, row in df.iterrows():
            df.loc[index,'datetime'] = start_datetime + timedelta(days = df.loc[index,'days_since'])
    elif i > 23 and i < 28:
        for index, row in df.iterrows():
            if np.isnan(df.loc[index,'days_since']):
                pass
            else:
                df.loc[index,'datetime'] = start_datetime_ERA5 + timedelta(hours = df.loc[index,'days_since'])  
        df = df.dropna()
    else:
        for index, row in df.iterrows():
                df.loc[index,'datetime'] = start_datetime_MERRA2 + timedelta(days = df.loc[index,'days_since'])
    df['binary'] = 1
    
    df.drop(columns=['days_since','AR_ID'], inplace = True)
    
    # Annual sum of ARs
    df.set_index('datetime', inplace = True)
    
    if sea_sep == True:
        year = 1979
        if i < 4:
            while year < 1999:
                df_temp = df.copy()
                for index, row in df_temp.iterrows():
                    if index.year != year:
                        df_temp.drop(index, axis = 0, inplace = True)
                    else:
                        pass
                df_temp['season'] = 0
                df_temp['season'] = df_temp.index.month%12//3+1
                df_temp = df_temp.groupby(['season'])['binary'].sum()
                
                fixer = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])
                df_temp = df_temp.add(fixer, fill_value=0)
                del(fixer)  
                
                df_sea_f19.iloc[0:4,count_f19] = df_temp[0:4]
                count_f19 = count_f19 + 1
                year = year + 1
            print("f19")
        elif i > 3 and i < 8:
            while year < 1999:
                df_temp = df.copy()
                for index, row in df_temp.iterrows():
                    if index.year != year:
                        try:
                            df_temp.drop(index, axis = 0, inplace = True)
                        except:
                            pass
                    else:
                        pass
                df_temp['season'] = df_temp.index.month%12//3+1
                df_temp = df_temp.groupby(['season'])['binary'].sum()
                
                fixer = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])
                df_temp = df_temp.add(fixer, fill_value=0)
                del(fixer)
                
                df_sea_f09.iloc[0:4,count_f09] = df_temp[0:4]
                count_f09 = count_f09 + 1
                year = year + 1
            print("f09")
        elif i > 7 and i < 12:
            while year < 1999:
                df_temp = df.copy()
                for index, row in df_temp.iterrows():
                    if index.year != year:
                        try:
                            df_temp.drop(index, axis = 0, inplace = True)
                        except:
                            pass
                    else:
                        pass
                df_temp['season'] = df_temp.index.month%12//3+1
                df_temp = df_temp.groupby(['season'])['binary'].sum()
                
                fixer = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])
                df_temp = df_temp.add(fixer, fill_value=0)
                del(fixer)
                
                df_sea_pg2.iloc[0:4,count_pg2] = df_temp[0:4]
                count_pg2 = count_pg2 + 1
                year = year + 1
            print("ne30pg2")
        elif i > 11 and i < 16:
            while year < 1999:
                df_temp = df.copy()
                for index, row in df_temp.iterrows():
                    if index.year != year:
                        try:
                            df_temp.drop(index, axis = 0, inplace = True)
                        except:
                            pass
                    else:
                        pass
                df_temp['season'] = df_temp.index.month%12//3+1
                df_temp = df_temp.groupby(['season'])['binary'].sum()
                
                fixer = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])
                df_temp = df_temp.add(fixer, fill_value=0)
                del(fixer)
                
                df_sea_pg3.iloc[0:4,count_pg3] = df_temp[0:4]
                count_pg3 = count_pg3 + 1
                year = year + 1
            print("ne30pg3")
        elif i > 15 and i < 20:
            while year < 1999:
                df_temp = df.copy()
                for index, row in df_temp.iterrows():
                    if index.year != year:
                        try:
                            df_temp.drop(index, axis = 0, inplace = True)
                        except:
                            pass
                    else:
                        pass
                df_temp['season'] = df_temp.index.month%12//3+1
                df_temp = df_temp.groupby(['season'])['binary'].sum()
                
                fixer = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])
                df_temp = df_temp.add(fixer, fill_value=0)
                del(fixer)
                
                df_sea_A.iloc[0:4,count_A] = df_temp[0:4]
                count_A = count_A + 1
                year = year + 1
            print("ARCTIC")
        elif i > 19 and i < 24:
            while year < 1999:
                df_temp = df.copy()
                for index, row in df_temp.iterrows():
                    if index.year != year:
                        try:
                            df_temp.drop(index, axis = 0, inplace = True)
                        except:
                            pass
                    else:
                        pass
                df_temp['season'] = df_temp.index.month%12//3+1
                df_temp = df_temp.groupby(['season'])['binary'].sum()
                
                fixer = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])
                df_temp = df_temp.add(fixer, fill_value=0)
                del(fixer) 
                
                df_sea_AG.iloc[0:4,count_AG] = df_temp[0:4]
                count_AG = count_AG + 1
                year = year + 1
            print("ARCTICGRIS")
        elif i > 23 and i < 28:
            while year < 1999:
                df_temp = df.copy()
                for index, row in df_temp.iterrows():
                    if index.year != year:
                        try:
                            df_temp.drop(index, axis = 0, inplace = True)
                        except:
                            pass
                    else:
                        pass
                df_temp['season'] = df_temp.index.month%12//3+1
                df_temp = df_temp.groupby(['season'])['binary'].sum()
                
                fixer = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])
                df_temp = df_temp.add(fixer, fill_value=0)
                del(fixer)
                
                df_sea_era5.iloc[0:4,count_era5] = df_temp[0:4]
                count_era5 = count_era5 + 1
                year = year + 1
            print("ERA5")
        else:
            year = 1980
            while year < 2000:
                df_temp = df.copy()
                for index, row in df_temp.iterrows():
                    if index.year != year:
                        try:
                            df_temp.drop(index, axis = 0, inplace = True)
                        except:
                            pass
                    else:
                        pass
                df_temp['season'] = df_temp.index.month%12//3+1
                df_temp = df_temp.groupby(['season'])['binary'].sum()
                
                fixer = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])
                df_temp = df_temp.add(fixer, fill_value=0)
                del(fixer)
                
                df_sea_merra2.iloc[0:4,count_merra2] = df_temp[0:4]
                count_merra2 = count_merra2 + 1
                year = year + 1
            print("MERRA2")
     
    i = i + 1

sea_dfs = [df_sea_f19, df_sea_f09, df_sea_pg2,df_sea_pg3,df_sea_A,df_sea_AG,df_sea_era5, df_sea_merra2]


# In[24]:


# Averaging dfs

# dataframes to hold seasonal info averaged by yer
df_sea_f19_abr = pd.DataFrame(index=years,columns=index_list)
df_sea_f09_abr =  pd.DataFrame(index=years,columns=index_list)
df_sea_pg2_abr = pd.DataFrame(index=years,columns=index_list)
df_sea_pg3_abr = pd.DataFrame(index=years,columns=index_list)
df_sea_A_abr = pd.DataFrame(index=years,columns=index_list)
df_sea_AG_abr = pd.DataFrame(index=years,columns=index_list)
df_sea_era5_abr = pd.DataFrame(index=years,columns=index_list)
df_sea_merra2_abr = pd.DataFrame(index=years_merra2,columns=index_list)
sea_abr_dfs = [df_sea_f19_abr, df_sea_f09_abr, df_sea_pg2_abr,df_sea_pg3_abr,df_sea_A_abr,df_sea_AG_abr,df_sea_era5_abr, df_sea_merra2_abr]



for df_abr, df_seasonal in zip(sea_abr_dfs, sea_dfs):
    #beginning column indices
    starts = [0,20,40,60]
    x = 0 # counter to iteratre through years, can go up to 18
    while x < 20:
        i = 0 # counter to iterate through seasons, can go up to 3
        while i < 4:
            df_abr.iloc[x,i] = df_seasonal.iloc[i,starts].mean()
            i = i + 1
            #print(i)
        #print(starts)
        starts = [sum(z) for z in zip(starts, [1,1,1,1])]  
        x = x + 1
        #print(x)

    print(df_abr)


# In[25]:


print("Seasonal trends: ")

for (df,name) in zip(sea_dfs,labels):
    i = 0
    print(name)
    while i < 4:
        print(i)
        test = mk.original_test(df.iloc[i])
        print(test)
        i = i + 1

print("Seasonal trends over year: ")
        
for (df,name) in zip(sea_abr_dfs,labels):
    i = 0
    print(name)
    while i < 4:
        test = mk.original_test(df.iloc[:,i])
        print(test)
        i = i + 1

print("Annual trends: ")

for (df,name) in zip(ann_dfs,labels):
    i = 0
    print(name)
    test = mk.original_test(df)
    print(test)
    i = i + 1
    


# In[30]:

medianprops = dict(linestyle='-', linewidth=4)

fig, ax = plt.subplots(figsize=(20,10))

df_sea_f19.fillna(0,inplace=True)
df_sea_f09.fillna(0,inplace=True)
df_sea_pg2.fillna(0,inplace=True)
df_sea_pg3.fillna(0,inplace=True)
df_sea_A.fillna(0,inplace=True)
df_sea_AG.fillna(0,inplace=True)
df_sea_era5.fillna(0,inplace=True)
df_sea_merra2.fillna(0,inplace=True)

ax.boxplot(df_sea_pg2.T, positions = [0.7,1.7,2.7,3.7], manage_ticks = False, widths = 0.1, patch_artist=True, boxprops=dict(facecolor=colors[2]), medianprops=medianprops)
ax.boxplot(df_sea_pg3.T, positions = [0.8,1.8,2.8,3.8], widths = 0.1, patch_artist=True, boxprops=dict(facecolor=colors[3]), medianprops=medianprops)
ax.boxplot(df_sea_f19.T, positions = [0.9,1.9,2.9,3.9], widths = 0.1, manage_ticks = False, patch_artist=True, boxprops=dict(facecolor=colors[0]), medianprops=medianprops)
ax.boxplot(df_sea_f09.T, positions = [1,2,3,4], manage_ticks = False, widths = 0.1, patch_artist=True, boxprops=dict(facecolor=colors[1]), medianprops=medianprops)
ax.boxplot(df_sea_A.T, positions = [1.1,2.1,3.1,4.1], manage_ticks = False, widths = 0.1, patch_artist=True, boxprops=dict(facecolor=colors[4]), medianprops=medianprops)
ax.boxplot(df_sea_AG.T, positions = [1.2,2.2,3.2,4.2], manage_ticks = False, widths = 0.1, patch_artist=True, boxprops=dict(facecolor=colors[5]), medianprops=medianprops)
ax.boxplot(df_sea_era5.T, positions = [1.3,2.3,3.3,4.3], manage_ticks = False, widths = 0.1, patch_artist=True, boxprops=dict(facecolor=colors[6]), medianprops=medianprops)
ax.boxplot(df_sea_merra2.T, positions = [1.4,2.4,3.4,4.4], manage_ticks = False, widths = 0.1, patch_artist=True, boxprops=dict(facecolor=colors[7]), medianprops=medianprops)


f19_patch = mpatches.Patch(color=colors[0], label=labels[0])
f09_patch = mpatches.Patch(color=colors[1], label=labels[1])
pg2_patch = mpatches.Patch(color=colors[2], label=labels[2])
pg3_patch = mpatches.Patch(color=colors[3], label=labels[3])
A_patch = mpatches.Patch(color=colors[4], label=labels[4])
AG_patch = mpatches.Patch(color=colors[5], label=labels[5])
era5_patch = mpatches.Patch(color=colors[6], label=labels[6])
merra2_patch = mpatches.Patch(color=colors[7], label=labels[7])

patches = [pg2_patch,pg3_patch,f19_patch,f09_patch,A_patch,AG_patch,era5_patch,merra2_patch]
plt.legend(handles=[pg2_patch,pg3_patch,f19_patch,f09_patch,A_patch,AG_patch,era5_patch,merra2_patch], loc="upper left", fontsize = 16, frameon=False)

fig.canvas.draw()

axis_names = [item.get_text() for item in ax.get_xticklabels()]
axis_names[0] = 'winter'
axis_names[1] = 'spring'
axis_names[2] = 'summer'
axis_names[3] = 'fall'

ax.set_xticklabels(axis_names)

plt.xticks([1.05,2.05,3.05,4.05], fontsize=25, labels = axis_names)
plt.yticks(fontsize=25)

plt.xlabel("season",fontsize = 30)
plt.ylabel("number of ARs", fontsize = 30)


plt.savefig("temp_seasonal.pdf",format = 'pdf')


# In[29]:


fig, ax = plt.subplots()

ax.plot(df_ann.index,df_ann['total'])

#plt.xlim(datetime(1980,1,1),datetime(1981,1,1))
#plt.ylim(0,40)

for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))
    
z = np.polyfit(df_ann.index, df_ann['total'], 1)
p = np.poly1d(z)

#add trendline to plot
ax.plot(df_ann.index, p(df_ann.index), color = 'red')
ci = 1.96 * np.std(df_ann['total'])/np.sqrt(len(df_ann.index))
ax.fill_between(df_ann.index, (df_ann['total']-ci), (df_ann['total']+ci), color='green', alpha=.1)

ci = 1.96 * np.std(p(df_ann.index))/np.sqrt(len(df_ann.index))
#ax.fill_between(df_ann.index, (p(df_ann.index)-ci), (p(df_ann.index)+ci), color='purple', alpha=.1)


plt.xlabel("Year")
plt.ylabel("Number ARs")
plt.title("Average number of ARs per year")


# In[27]:


fig, ax = plt.subplots(2,1, sharex = True)

plt.xticks(np.arange(1979,1998,step = 2))

i = 0
while i < 7:
    ax[0].plot(ann_dfs[i].index,ann_dfs[i]['total'],color=colors[i], label = labels[i])
    ci = 1.96 * np.std(ann_dfs[i]['total'])/np.sqrt(len(ann_dfs[i].index))
    ax[0].fill_between(ann_dfs[i].index, (ann_dfs[i]['total']-ci), (ann_dfs[i]['total']+ci), color=colors[i], alpha=.1)
    
    
    z = np.polyfit(ann_dfs[i].index, ann_dfs[i]['total'], 1)
    p = np.poly1d(z)

    #add trendline to plot
    ax[1].plot(ann_dfs[i].index, p(ann_dfs[i].index), color = colors[i], linestyle = 'dashed')
    i = i + 1
    
ax[0].set_title("Average number of ARs per year")
ax[1].set_title("Linear trend of annual AR counts")

ax[1].set_xlabel("Year")

ax[0].set_ylabel("Number ARs")
ax[1].set_ylabel("Number ARs")

ax[0].legend(loc = 'best', bbox_to_anchor=(1, 0.40), fontsize = 10)

plt.xticks(np.arange(1979,1998,step = 2))


# In[28]:


fig, ax = plt.subplots(figsize=(8,4))

plt.xticks(np.arange(1979,1999,step = 2))

i = 0
while i < 8:
    ax.plot(ann_dfs[i].index,ann_dfs[i]['total'],color=colors[i], label = labels[i])

    i = i + 1
    

ax.set_xlabel("Year")

ax.set_ylabel("Number ARs")

ax.legend(loc = 'best', bbox_to_anchor=(1, 0.70), fontsize = 10)

plt.xticks(np.arange(1979,1999,step = 2))

fig.savefig("annualARs.pdf",format = 'pdf', bbox_inches='tight')

#plt.show()


# In[ ]:





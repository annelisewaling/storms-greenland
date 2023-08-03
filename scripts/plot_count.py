#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:46:38 2023

@author: alpac
"""
#%%
import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns
#from matplotlib import rc
#rc('mathtext', default='regular')

#%%

dfARs = pd.read_csv('ARclimatology.csv',delimiter=",",comment='#',header=0)

dfARs.set_index(dfARs['gridname'], drop = True, inplace = True)

dfARs.drop(columns=['gridname'], inplace = True)

#%% plot ARs

sns.set_style('dark') # darkgrid, white grid, dark, white and ticks
# plt.rc('axes', titlesize=16)     # fontsize of the axes title
# plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
# plt.rc('legend', fontsize=13)    # legend fontsize
# plt.rc('font', size=13)          # controls default text sizes

#sns.set_style()

fig = plt.figure(figsize=(6,4), tight_layout=True)# Create matplotlib figure

ax = fig.add_subplot(111)

ax2 = ax.twinx()

#ax2 = dfARs.iloc[:,2].plot.bar(rot=0, color='orange')

width = 0.2

dfARs.globalARs.plot(kind='bar', color='#377eb8', ax=ax, width=width, position=1)
dfARs.perc.plot(kind='bar', color='#4daf4a', ax=ax2, width=width, position=0)

ax.set_ylabel('Northern Hemisphere AR Count')
ax2.set_ylabel('% ARs Intersecting GrIS')

#ax2.xaxis.set_major_locator(5)
#ax2.xaxis.set_minor_locator(xminor_locator)
ax.set_ylim(0,12000)
ax2.set_ylim(0,6)


mylabels = ['Northern Hemisphere ARs','% ARs intersecting GrIS']

# ask matplotlib for the plotted objects and their labels
lines, labels = ('Northern Hemisphere ARs',ax)
lines2, labels2 = ('% ARs intersecting GrIS',ax2)
#lines2, labels2 = ax2.get_legend_handles_labels()
#ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.2, -0.2))
fig.legend(labels=mylabels,bbox_to_anchor=(0.39, 0.15))
fancybox=True

#ax.legend(title='Statistic', title_fontsize='13', loc='upper right')

plt.savefig('statistics.pdf',format='pdf', dpi=1500)




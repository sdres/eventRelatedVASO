'''

Plotting figures for the manuscript

'''


import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt


from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

plt.style.use('dark_background')

# set general styles

#########################################
########## set general styles ##########
#########################################

# define figzize
FS = (8, 5)

# define linewidth to 2
LW = 2

# define fontsize size for x- and y-labels
labelSize = 24

# define fontsize size for x- and y-ticks
tickLabelSize = 18

# define fontsize legend text

legendTextSize = 18

TR = 1.3



####################################################
########## Plot tSNR for short vs long TR ##########
####################################################

data = pd.read_csv('../results/shortVsLongTRData.csv')

palette = {
    'blockStimLongTR': '#8DD3C7',
    'blockStim': '#FFFFB3'}

labels = ['Short TR','Long TR']

# Plotting everything together
for i, modality in enumerate(['BOLD','VASO']):

    fig, ax = plt.subplots(figsize=(6,6))

    for j, length in enumerate(['blockStim', 'blockStimLongTR']):

        sns.kdeplot(
            ax=ax,
            data=data.loc[(data['modality']==modality)&(data['TRlength']==length)],
            x='tSNR',
            linewidth=2,
            color = palette[length],
            label=labels[j]
            )

    ax.set_yticks([])
    ax.set_ylabel('Density',fontsize=20)
    ax.set_xlim(0,45)

    # adapt x-axis
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_xlabel('tSNR',fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.set_xticks(range(0,46,5))


    ax.legend(loc='upper right', fontsize=tickLabelSize)

    plt.savefig(f'../results/v1_{modality}_longVsShort.jpg',bbox_inches='tight')
    plt.show()


########################################################
########## Plot profiles for short vs long TR ##########
########################################################

zscores = pd.read_csv('../results/blocksVsEventsData.csv')
zscores = zscores.loc[(zscores['subject']=='sub-09')|(zscores['subject']=='sub-11')]
zscores = zscores.loc[(zscores['stimType']=='blockStim')|(zscores['stimType']=='blockStimLongTR')]

for i, modality in enumerate(['BOLD','VASO']):
    fig, ax = plt.subplots(figsize=FS)

    for j, length in enumerate(['blockStim', 'blockStimLongTR']):

        sns.lineplot(
            ax=ax,
            data=zscores.loc[(zscores['modality']==modality)&(zscores['stimType']==length)],
            x='layer',
            y='data',
            ci=False,
            linewidth=2,
            color = palette[length],
            label=labels[j]
            )

    if modality == 'VASO':
        ax.set_ylim(0,5)
    if modality == 'BOLD':
        ax.set_ylim(0,10)

    plt.ylabel(f'Z-score',
        fontsize=labelSize
        )

    plt.xlabel('Cortical depth',
        fontsize = labelSize
        )

    ax.set_xticks([1,11],['WM', 'CSF'],
        fontsize = tickLabelSize
        )


    plt.yticks(fontsize = tickLabelSize)

    plt.legend(loc = 'upper left',
        fontsize = legendTextSize
        )
    plt.title(modality, fontsize=36, pad=20)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:02d}'.format(int(x)) for x in current_values])

    plt.savefig(f'../results/{modality}_profiles_v1longVsShort.jpg',bbox_inches='tight')
    plt.show()


###########################################
########## Plot block timcourses ##########
###########################################

data = pd.read_csv('../results/blockTimecourseResults.csv')

fig, ax = plt.subplots(figsize=FS)

sns.lineplot(ax=ax, data=data.loc[(data['focus']=='v1')], x='x', y='data', hue='modality', palette = palette,linewidth=LW)


ax.axvspan(4, 4+(30/TR), color='grey', alpha=0.2, lw=0)
ax.set_ylabel('Signal change [%]', fontsize=labelSize)
ax.set_xlabel('Time [s]', fontsize=labelSize)
ax.legend(loc='lower center', fontsize=tickLabelSize)

values = (np.arange(-4,len(data['x'].unique())-4,4)*TR).round(decimals=1)
spacing = np.arange(0,len(data['x'].unique()),4)
ax.set_xticks(spacing,values, fontsize=tickLabelSize)

ax.tick_params(axis='y', labelsize=tickLabelSize)
ax.axhline(0,linestyle='--',color='white')
plt.savefig('../results/blockTimecourseResults.png',
    bbox_inches = 'tight')
plt.show()


#########################################
########## Plot block profiles ##########
#########################################

# load data
zscores = pd.read_csv('../results/blocksVsEventsData.csv')
# Limit plotting to modality, focus and visiotactile stimulation and remove
# long TR block stimulation for fair comparisons
tmp = zscores.loc[
    (zscores['runType']=='blockStim')
    & (zscores['focus']=='v1')
    & (zscores['contrast']=='visiotactile')
    & (zscores['stimType']!='blockStimLongTR')
    ]

palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'
    }


fig, ax = plt.subplots(figsize=FS)

sns.lineplot(ax=ax,
    data=tmp,
    x='layer',
    y='data',
    hue='modality',
    palette=palette,
    linewidth = LW
    )


plt.ylabel(f'Z-score',
    fontsize=labelSize
    )

plt.xlabel('Cortical depth',
    fontsize = labelSize
    )


ax.set_xticks([1,11],['WM', 'CSF'],
    fontsize = tickLabelSize
    )

# Remove ticks for x-axis
# plt.xticks([])

ticks = np.arange(0,19,3)
plt.yticks(ticks,fontsize = tickLabelSize)

plt.legend(loc = 'upper left',
    fontsize = legendTextSize
    )

# current_values = plt.gca().get_yticks()
# plt.gca().set_yticklabels(['{:02d}'.format(x) for x in current_values])


plt.savefig(f'../results/Group_v1_blockProfiles.png',
    bbox_inches = 'tight')

plt.show()


#####################################
########## Plot event FIRs ##########
#####################################

data = pd.read_csv('../results/FIR_results.csv')
palettesLayers = {'VASO':['#1f77b4','#7dadd9','#c9e5ff'],
'BOLD':['#ff7f0e', '#ffae6f','#ffdbc2']}


for i, modality in enumerate(['BOLD', 'VASO']):

    fig, ax = plt.subplots(figsize=FS)

    tmp = data.loc[(data['modality']==modality)&(data['focus']=='v1')]

    sns.lineplot(ax=ax, data=tmp , x="volume", y="data", hue='layer',palette=palettesLayers[modality],linewidth=2)


    yLimits = ax.get_ylim()
    ax.set_ylim(-2,9)
    ax.set_yticks(range(-2,10,2),fontsize=18)

    # prepare x-ticks
    ticks = range(1,12,2)
    labels = (np.arange(0,11,2)*1.3).round(decimals=1)
    for k,label in enumerate(labels):
        if (label - int(label) == 0):
            labels[k] = int(label)

    ax.yaxis.set_tick_params(labelsize=tickLabelSize)
    ax.xaxis.set_tick_params(labelsize=tickLabelSize)

    if i == 0:
        ax.set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)
    else:
        ax.set_ylabel(r'', fontsize=labelSize)
        ax.set_yticks([])

    ax.legend(loc='upper right',fontsize=tickLabelSize)

    # tweak x-axis
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels,
        fontsize=tickLabelSize)
    ax.set_xlabel('Time [s]',
        fontsize=labelSize)
    ax.set_title(modality, fontsize=36)

    # draw lines
    ax.axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
    ax.axhline(0,linestyle='--',color='white')


    # ax.axvline((np.arange(0,11)*1.3).round(decimals=1)[3],linestyle='--',color='white')

    plt.savefig(f'../results/groupLevel_{focus}_{modality}_eventResults_withLayers.png', bbox_inches = "tight")


    plt.show()



#########################################################
########## Plot event FIRs for two individuals ##########
#########################################################

data = pd.read_csv('../results/FIR_results.csv')
palettesLayers = {'VASO':['#1f77b4','#7dadd9','#c9e5ff'],
'BOLD':['#ff7f0e', '#ffae6f','#ffdbc2']}

limits = {'sub-05':16,
    'sub-08':12,
    'sub-14':9
    }

for sub in ['sub-05', 'sub-08', 'sub-14']:
    for i, modality in enumerate(['BOLD', 'VASO']):

        fig, ax = plt.subplots(figsize=(7,5))

        tmp = data.loc[(data['subject']==sub)&(data['modality']==modality)&(data['focus']=='v1')]

        sns.lineplot(ax=ax, data=tmp , x="volume", y="data", hue='layer',palette=palettesLayers[modality],linewidth=2)


        yLimits = ax.get_ylim()
        ax.set_ylim(-2,limits[sub])
        ax.set_yticks(range(-2,limits[sub]+1,2),fontsize=18)

        # prepare x-ticks
        ticks = range(1,12,2)
        labels = (np.arange(0,11,2)*1.3).round(decimals=1)
        for k,label in enumerate(labels):
            if (label - int(label) == 0):
                labels[k] = int(label)

        ax.yaxis.set_tick_params(labelsize=tickLabelSize)
        ax.xaxis.set_tick_params(labelsize=tickLabelSize)

        if i == 0:
            ax.set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)
        else:
            ax.set_ylabel(r'', fontsize=labelSize)
            ax.set_yticks([])

        ax.legend(loc='upper right',fontsize=tickLabelSize)

        # tweak x-axis
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels,
            fontsize=tickLabelSize)
        ax.set_xlabel('Time [s]',
            fontsize=labelSize)
        ax.set_title(modality, fontsize=36)

        # draw lines
        ax.axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
        ax.axhline(0,linestyle='--',color='white')

        # ax.axvline((np.arange(0,11)*1.3).round(decimals=1)[3],linestyle='--',color='white')

        plt.savefig(f'../results/{sub}_v1_{modality}_eventResults_withLayers.png', bbox_inches = "tight")


        plt.show()


#########################################
########## Plot event profiles ##########
#########################################

# load data
zscores = pd.read_csv('../results/blocksVsEventsData.csv')
# Limit plotting to modality, focus and visiotactile stimulation and remove
# long TR block stimulation for fair comparisons
tmp = zscores.loc[
    (zscores['runType']=='eventStim')
    & (zscores['focus']=='v1')
    & (zscores['contrast']=='visiotactile')
    & (zscores['stimType']!='blockStimLongTR')
    ]

palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'
    }


fig, ax = plt.subplots(figsize=FS)

sns.lineplot(ax=ax,
    data=tmp,
    x='layer',
    y='data',
    hue='modality',
    palette=palette,
    linewidth = LW
    )


plt.ylabel(f'Z-score',
    fontsize=labelSize
    )

plt.xlabel('Cortical depth',
    fontsize = labelSize
    )


ax.set_xticks([1,11],['WM', 'CSF'],
    fontsize = tickLabelSize
    )

# Remove ticks for x-axis
# plt.xticks([])

ticks = np.arange(0,13,2)
plt.yticks(ticks,fontsize = tickLabelSize)

plt.legend(loc = 'upper left',
    fontsize = legendTextSize
    )


plt.savefig(f'../results/Group_v1_eventProfiles.png',
    bbox_inches = 'tight')

plt.show()


###########################################
########## Plot blocks vs events ##########
###########################################

# load data
zscores = pd.read_csv('../results/blocksVsEventsData.csv')

# Define two colors for each modality
VASOcmap = {
    'eventStim': '#7dadd9',
    'blockStim': '#1f77b4'}
BOLDcmap = {
    'eventStim': '#ffae6f',
    'blockStim': '#ff7f0e'}

# store as list to loop over
palettes = [VASOcmap,BOLDcmap]



for modality, cmap in zip(['VASO', 'BOLD'], palettes):

    fig, ax = plt.subplots(figsize=FS)

    # Limit plotting to modality, focus and visiotactile stimulation and remove
    # long TR block stimulation for fair comparisons
    tmp = zscores.loc[
        (zscores['modality']==modality)
        & (zscores['focus']=='v1')
        & (zscores['contrast']=='visiotactile')
        & (zscores['stimType']!='blockStimLongTR')
        ]

    sns.lineplot(ax=ax,
        data=tmp,
        x='layer',
        y='data',
        hue='runType',
        palette=cmap,
        linewidth = LW
        )


    plt.ylabel(f'Z-score',
        fontsize=labelSize
        )

    plt.xlabel('Cortical depth',
        fontsize = labelSize
        )


    ax.set_xticks([1,11],['WM', 'CSF'],
        fontsize = tickLabelSize
        )

    # Remove ticks for x-axis
    # plt.xticks([])


    plt.yticks(fontsize = tickLabelSize)

    # Set y-limits for modalities
    if modality == 'VASO':
        ticks = np.arange(0,7,1)
        plt.yticks(ticks,fontsize = tickLabelSize)
    if modality == 'BOLD':
        ticks = np.arange(0,19,3)
        plt.yticks(ticks,fontsize = tickLabelSize)


    plt.legend(loc = 'upper left',
        fontsize = legendTextSize
        )


    plt.savefig(f'../results/Group_v1_{modality}_blocksVsEvents.png',
        bbox_inches = 'tight')

    plt.show()



##########################################
########## Plot efficiency data ##########
##########################################


data = pd.read_csv('../results/efficiencyData.csv')
data = data.loc[(data['focus']=='v1')]

palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'}


fig, ax = plt.subplots(figsize=FS)


sns.lineplot(ax= ax,
    data=data,
    x='nrTrials',
    y='score',
    hue='modality',
    palette= palette,
    linewidth=LW
    )

# Set labels and sizes
ax.set_ylabel('Error',
    fontsize=labelSize)
ax.set_xlabel('Averaged trials',
    fontsize=labelSize)


ax.yaxis.set_tick_params(labelsize=tickLabelSize)
ax.xaxis.set_tick_params(labelsize=tickLabelSize)

ax.set_ylim(0,25)

plt.legend(loc = 'upper right',
    fontsize = legendTextSize
    )


ax.vlines(20,
    ymin=0,
    ymax=30,
    label='5% error',
    linestyle='dashed',
    color='white'
    )


# fig.tight_layout()
plt.savefig(f'../results/Group_v1_stabilizing.png',
    bbox_inches = 'tight')

plt.show()


############################################
########## Plot event FIRs random ##########
############################################

data = pd.read_csv('../results/FIRdataRandom.csv')

# palettesLayers = {'VASO':['#1f77b4','#7dadd9','#c9e5ff'],
# 'BOLD':['#ff7f0e', '#ffae6f','#ffdbc2']}

VASOcmap = {
    'visual': '#7dadd9',
    'visiotactile': '#1f77b4'}
BOLDcmap = {
    'visual': '#ffae6f',
    'visiotactile': '#ff7f0e'}

# store as list to loop over
palettes = [BOLDcmap,VASOcmap]


for j, modality in enumerate(['BOLD', 'VASO']):
    fig, axes = plt.subplots(1,3, figsize=(10,5))
    fig.subplots_adjust(top=0.8)



    for i, layer in enumerate(['superficial', 'middle', 'deep']):


        tmp = data.loc[(data['modality']==modality)&(data['layer']==layer)]

        sns.lineplot(ax=axes[i], data=tmp, x='volume', y='data', hue='condition',linewidth=LW, palette=palettes[j])


        if modality == 'BOLD':
            # axes[i].set_ylim(-1,7)
            axes[i].set_yticks(range(-1,8,2),fontsize=tickLabelSize)
        if modality == 'VASO':
            # axes[i].set_ylim(-1,4)
            axes[i].set_yticks(range(-1,5,1),fontsize=tickLabelSize)

        # prepare x-ticks
        ticks = range(1,12,3)
        labels = (np.arange(0,11,3)*1.3).round(decimals=1)
        for k,label in enumerate(labels):
            if (label - int(label) == 0):
                labels[k] = int(label)

        axes[i].yaxis.set_tick_params(labelsize=tickLabelSize)
        axes[i].xaxis.set_tick_params(labelsize=tickLabelSize)

        if i == 0:
            axes[i].set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)
        else:
            axes[i].set_ylabel(r'', fontsize=labelSize)
            axes[i].set_yticks([])


        # tweak x-axis
        axes[i].set_xticks(ticks)
        axes[i].set_xticklabels(labels,
            fontsize=tickLabelSize)
        axes[i].set_xlabel('Time [s]',
            fontsize=labelSize)
        axes[i].set_title(layer, fontsize=labelSize)

        # draw lines
        axes[i].axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0)
        axes[i].axhline(0,linestyle='--',color='white')

    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].legend(loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize = legendTextSize
        )

    plt.suptitle(f'{modality}', fontsize=labelSize, y=0.98)
    plt.savefig(f'../results/FIR_{modality}_visuotactile.png', bbox_inches = "tight")
    plt.show()





################################################################################
####################################   S1   ####################################
################################################################################



###########################################
########## Plot block timcourses ##########
###########################################

data = pd.read_csv('../results/blockTimecourses.csv')

fig, ax = plt.subplots(figsize=FS)

sns.lineplot(ax=ax, data=data.loc[(data['focus']=='s1')], x='x', y='data', hue='modality', palette = palette,linewidth=LW)


ax.axvspan(4, 4+(30/TR), color='grey', alpha=0.2, lw=0)
ax.set_ylabel('Signal change [%]', fontsize=labelSize)
ax.set_xlabel('Time [s]', fontsize=labelSize)
ax.legend(loc='lower center', fontsize=tickLabelSize)

values = (np.arange(-4,len(data['x'].unique())-4,4)*TR).round(decimals=1)
spacing = np.arange(0,len(data['x'].unique()),4)
ax.set_xticks(spacing,values, fontsize=tickLabelSize)

ax.tick_params(axis='y', labelsize=tickLabelSize)
ax.axhline(0,linestyle='--',color='white')
plt.savefig('../results/blockTimecourseResults_s1.png',
    bbox_inches = 'tight')
plt.show()


#########################################
########## Plot block profiles ##########
#########################################

# load data
zscores = pd.read_csv('../results/blocksVsEventsData.csv')
# Limit plotting to modality, focus and visiotactile stimulation and remove
# long TR block stimulation for fair comparisons
tmp = zscores.loc[
    (zscores['runType']=='blockStim')
    & (zscores['focus']=='s1')
    & (zscores['contrast']=='visiotactile')
    & (zscores['stimType']!='blockStimLongTR')
    ]

palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'
    }


fig, ax = plt.subplots(figsize=FS)

sns.lineplot(ax=ax,
    data=tmp,
    x='layer',
    y='data',
    hue='modality',
    palette=palette,
    linewidth = LW
    )


plt.ylabel(f'Z-score',
    fontsize=labelSize
    )

plt.xlabel('Cortical depth',
    fontsize = labelSize
    )


ax.set_xticks([1,11],['WM', 'CSF'],
    fontsize = tickLabelSize
    )

# Remove ticks for x-axis
# plt.xticks([])

ticks = np.arange(0,11,2)
plt.yticks(ticks, fontsize = tickLabelSize)

plt.legend(loc = 'upper left',
    fontsize = legendTextSize
    )

plt.savefig(f'../results/Group_s1_blockProfiles.png',
    bbox_inches = 'tight')

plt.show()


#####################################
########## Plot event FIRs ##########
#####################################

data = pd.read_csv('../results/FIR_results.csv')
palettesLayers = {'VASO':['#1f77b4','#7dadd9','#c9e5ff'],
'BOLD':['#ff7f0e', '#ffae6f','#ffdbc2']}

for focus in ['v1','s1']:
    for i, modality in enumerate(['BOLD', 'VASO']):

        fig, ax = plt.subplots(figsize=FS)

        tmp = data.loc[(data['modality']==modality)&(data['focus']==focus)]

        sns.lineplot(ax=ax, data=tmp , x="volume", y="data", hue='layer',palette=palettesLayers[modality],linewidth=2)


        yLimits = ax.get_ylim()
        ax.set_ylim(-2,9)
        ax.set_yticks(range(-2,10,2),fontsize=18)

        # prepare x-ticks
        ticks = range(1,12,2)
        labels = (np.arange(0,11,2)*1.3).round(decimals=1)
        for k,label in enumerate(labels):
            if (label - int(label) == 0):
                labels[k] = int(label)

        ax.yaxis.set_tick_params(labelsize=tickLabelSize)
        ax.xaxis.set_tick_params(labelsize=tickLabelSize)

        if i == 0:
            ax.set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)
        else:
            ax.set_ylabel(r'', fontsize=labelSize)
            ax.set_yticks([])

        ax.legend(loc='upper right',fontsize=tickLabelSize)

        # tweak x-axis
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels,
            fontsize=tickLabelSize)
        ax.set_xlabel('Time [s]',
            fontsize=labelSize)
        ax.set_title(modality, fontsize=36)

        # draw lines
        ax.axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
        ax.axhline(0,linestyle='--',color='white')


        # ax.axvline((np.arange(0,11)*1.3).round(decimals=1)[3],linestyle='--',color='white')

        plt.savefig(f'../results/groupLevel_{focus}_{modality}_eventResults_withLayers.png', bbox_inches = "tight")


        plt.show()

#########################################
########## Plot event profiles ##########
#########################################

# load data
zscores = pd.read_csv('../results/blocksVsEventsData.csv')
# Limit plotting to modality, focus and visiotactile stimulation and remove
# long TR block stimulation for fair comparisons

for focus in ['s1']:

    tmp = zscores.loc[
        (zscores['runType']=='eventStim')
        & (zscores['focus']==focus)
        & (zscores['contrast']=='visiotactile')
        & (zscores['stimType']!='blockStimLongTR')
        ]

    palette = {
        'BOLD': 'tab:orange',
        'VASO': 'tab:blue'
        }


    fig, ax = plt.subplots(figsize=FS)

    sns.lineplot(ax=ax,
        data=tmp,
        x='layer',
        y='data',
        hue='modality',
        palette=palette,
        linewidth = LW
        )


    plt.ylabel(f'Z-score',
        fontsize=labelSize
        )

    plt.xlabel('Cortical depth',
        fontsize = labelSize
        )


    ax.set_xticks([1,11],['WM', 'CSF'],
        fontsize = tickLabelSize
        )

    # Remove ticks for x-axis
    # plt.xticks([])

    ticks = np.arange(0,11,2)
    plt.yticks(ticks,fontsize = tickLabelSize)

    plt.legend(loc = 'upper left',
        fontsize = legendTextSize
        )


    plt.savefig(f'../results/Group_{focus}_eventProfiles.png',
        bbox_inches = 'tight')

    plt.show()

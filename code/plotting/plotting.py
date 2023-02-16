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

data = pd.read_csv('../../results/backupOld/shortVsLongTRData.csv')

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

zscores = pd.read_csv('../../results/backupOld/blocksVsEventsData.csv')
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

    ax.set_xticks([1, 11], ['WM', 'CSF'], fontsize = tickLabelSize)

    plt.yticks(fontsize=tickLabelSize)

    plt.legend(loc='upper left', fontsize=legendTextSize)
    plt.title(modality, fontsize=36, pad=20)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:02d}'.format(int(x)) for x in current_values])

    plt.savefig(f'../results/{modality}_profiles_v1longVsShort.jpg',bbox_inches='tight')
    plt.show()

###########################################
########## Plot block timcourses ##########
###########################################

data = pd.read_csv('../../results/backupOld/blockTimecourseResults.csv')

fig, ax = plt.subplots(figsize=FS)

sns.lineplot(ax=ax, data=data.loc[(data['focus'] == 'v1')], x='x', y='data', hue='modality', palette=palette, linewidth=LW)


ax.axvspan(4, 4+(30/TR), color='grey', alpha=0.2, lw=0)
ax.set_ylabel('Signal change [%]', fontsize=labelSize)
ax.set_xlabel('Time [s]', fontsize=labelSize)
ax.legend(loc='lower center', fontsize=tickLabelSize)

values = (np.arange(-4,len(data['x'].unique())-4, 4)*TR).round(decimals=1)
spacing = np.arange(0,len(data['x'].unique()), 4)
ax.set_xticks(spacing, values, fontsize=tickLabelSize)

ax.tick_params(axis='y', labelsize=tickLabelSize)
ax.axhline(0, linestyle='--', color='white')
plt.savefig('../results/blockTimecourseResults.png', bbox_inches='tight')
plt.show()

#########################################
########## Plot block profiles ##########
#########################################

# load data
zscores = pd.read_csv('../../results/backupOld/blocksVsEventsData.csv')
# Limit plotting to modality, focus and visiotactile stimulation and remove
# long TR block stimulation for fair comparisons
tmp = zscores.loc[
    (zscores['runType']=='blockStim')
    & (zscores['focus']=='v1')
    & (zscores['contrast']=='visiotactile')
    & (zscores['stimType']!='blockStimLongTR')
    ]

palette = {'BOLD': 'tab:orange',
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


plt.ylabel(f'Z-score', fontsize=labelSize)

plt.xlabel('Cortical depth', fontsize=labelSize)

ax.set_xticks([1,11],['WM', 'CSF'], fontsize=tickLabelSize)

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

####################################################
########## Plot event FIRs without layers ##########
####################################################

data = pd.read_csv('../../results/backupOld/FIR_results.csv')

palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'
    }

fig, ax = plt.subplots(figsize=FS)

tmp = data.loc[(data['focus']=='v1')]

sns.lineplot(ax=ax, data=tmp , x="volume", y="data", hue='modality', palette=palette, linewidth=2)


yLimits = ax.get_ylim()
# ax.set_ylim(-1,7)
ax.set_yticks(range(-1,7),fontsize=18)

# prepare x-ticks
ticks = range(1,12,2)
labels = (np.arange(0,11,2)*1.3).round(decimals=1)
for k,label in enumerate(labels):
    if (label - int(label) == 0):
        labels[k] = int(label)

ax.yaxis.set_tick_params(labelsize=tickLabelSize)
ax.xaxis.set_tick_params(labelsize=tickLabelSize)

ax.set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)


ax.legend(loc='upper right',fontsize=tickLabelSize)

# tweak x-axis
ax.set_xticks(ticks)
ax.set_xticklabels(labels,
    fontsize=tickLabelSize)
ax.set_xlabel('Time [s]',
    fontsize=labelSize)
# draw lines
ax.axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
ax.axhline(0,linestyle='--',color='white')


# ax.axvline((np.arange(0,11)*1.3).round(decimals=1)[3],linestyle='--',color='white')

plt.savefig(f'../results/groupLevel_v1_eventResults_withoutLayers.png', bbox_inches="tight")


plt.show()


#####################################
########## Plot event FIRs ##########
#####################################

data = pd.read_csv('../../results/backupOld/FIR_results.csv')
palettesLayers = {'VASO':['#1f77b4','#7dadd9','#c9e5ff'],
'BOLD':['#ff7f0e', '#ffae6f','#ffdbc2']}


for i, modality in enumerate(['BOLD', 'VASO']):

    fig, ax = plt.subplots(figsize=FS)

    tmp = data.loc[(data['modality'] == modality) & (data['focus'] == 'v1')]

    sns.lineplot(ax=ax, data=tmp, x="volume", y="data", hue='layer', palette=palettesLayers[modality], linewidth=2)


    yLimits = ax.get_ylim()
    ax.set_ylim(-2, 9)
    ax.set_yticks(range(-2, 10, 2), fontsize=18)

    # prepare x-ticks
    ticks = range(1, 12, 2)
    labels = (np.arange(0, 11, 2)*1.3).round(decimals=1)
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

    ax.legend(loc='upper right', fontsize=tickLabelSize)

    # tweak x-axis
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels,
        fontsize=tickLabelSize)
    ax.set_xlabel('Time [s]',
        fontsize=labelSize)
    ax.set_title(modality, fontsize=36)

    # draw lines
    ax.axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation on')
    ax.axhline(0, linestyle='--', color='white')


    # ax.axvline((np.arange(0,11)*1.3).round(decimals=1)[3],linestyle='--',color='white')

    plt.savefig(f'../results/groupLevel_{focus}_{modality}_eventResults_withLayers.png', bbox_inches="tight")


    plt.show()



#########################################################
########## Plot event FIRs for two individuals ##########
#########################################################

data = pd.read_csv('../../results/backupOld/FIR_results.csv')
palettesLayers = {'VASO': ['#1f77b4', '#7dadd9', '#c9e5ff'],
                  'BOLD': ['#ff7f0e', '#ffae6f', '#ffdbc2']
                  }

limits = {'sub-05': 16,
          'sub-08': 12,
          'sub-14': 9
          }

for sub in ['sub-05', 'sub-08', 'sub-14']:
    for i, modality in enumerate(['BOLD', 'VASO']):

        fig, ax = plt.subplots(figsize=(7, 5))

        tmp = data.loc[(data['subject']==sub)&(data['modality'] == modality)&(data['focus'] == 'v1')]

        sns.lineplot(ax=ax, data=tmp, x="volume", y="data", hue='layer', palette=palettesLayers[modality], linewidth=2)


        yLimits = ax.get_ylim()
        ax.set_ylim(-2, limits[sub])
        ax.set_yticks(range(-2, limits[sub]+1, 2), fontsize=18)

        # prepare x-ticks
        ticks = range(1, 12, 2)
        labels = (np.arange(0, 11, 2)*1.3).round(decimals=1)
        for k, label in enumerate(labels):
            if (label - int(label) == 0):
                labels[k] = int(label)

        ax.yaxis.set_tick_params(labelsize=tickLabelSize)
        ax.xaxis.set_tick_params(labelsize=tickLabelSize)

        if i == 0:
            ax.set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)
        else:
            ax.set_ylabel(r'', fontsize=labelSize)
            ax.set_yticks([])

        ax.legend(loc='upper right', fontsize=tickLabelSize)

        # tweak x-axis
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=tickLabelSize)
        ax.set_xlabel('Time [s]', fontsize=labelSize)
        ax.set_title(modality, fontsize=36)

        # draw lines
        ax.axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation on')
        ax.axhline(0, linestyle='--', color='white')

        # ax.axvline((np.arange(0,11)*1.3).round(decimals=1)[3],linestyle='--',color='white')

        plt.savefig(f'../results/{sub}_v1_{modality}_eventResults_withLayers.png', bbox_inches = "tight")


        plt.show()


#########################################
########## Plot event profiles ##########
#########################################

# load data
zscores = pd.read_csv('../../results/backupOld/blocksVsEventsData.csv')
# Limit plotting to modality, focus and visiotactile stimulation and remove
# long TR block stimulation for fair comparisons
tmp = zscores.loc[(zscores['runType']=='eventStim')
                  & (zscores['focus']=='v1')
                  & (zscores['contrast']=='visiotactile')
                  & (zscores['stimType']!='blockStimLongTR')
]

palette = {'BOLD': 'tab:orange',
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


plt.ylabel(f'Z-score', fontsize=labelSize)

plt.xlabel('Cortical depth', fontsize=labelSize)

ax.set_xticks([1,11],['WM', 'CSF'], fontsize=tickLabelSize)

# Remove ticks for x-axis
# plt.xticks([])

ticks = np.arange(0, 13, 2)
plt.yticks(ticks, fontsize=tickLabelSize)

plt.legend(loc='upper left', fontsize=legendTextSize)


plt.savefig(f'../results/Group_v1_eventProfiles.png', bbox_inches='tight')

plt.show()


###########################################
########## Plot blocks vs events ##########
###########################################

# load data
zscores = pd.read_csv('../../results/backupOld/blocksVsEventsData.csv')

# Define two colors for each modality
VASOcmap = {'eventStim': '#7dadd9',
            'blockStim': '#1f77b4'
            }

BOLDcmap = {'eventStim': '#ffae6f',
            'blockStim': '#ff7f0e'
            }

# store as list to loop over
palettes = [VASOcmap,BOLDcmap]

for modality, cmap in zip(['VASO', 'BOLD'], palettes):

    fig, ax = plt.subplots(figsize=FS)

    # Limit plotting to modality, focus and visiotactile stimulation and remove
    # long TR block stimulation for fair comparisons
    tmp = zscores.loc[(zscores['modality'] == modality)
                      & (zscores['focus'] == 'v1')
                      & (zscores['contrast'] == 'visiotactile')
                      & (zscores['stimType'] != 'blockStimLongTR')
    ]

    sns.lineplot(ax=ax,
                 data=tmp,
                 x='layer',
                 y='data',
                 hue='runType',
                 palette=cmap,
                 linewidth = LW
                 )


    plt.ylabel(f'Z-score', fontsize=labelSize)

    plt.xlabel('Cortical depth', fontsize=labelSize)


    ax.set_xticks([1, 11], ['WM', 'CSF'], fontsize=tickLabelSize)

    # Remove ticks for x-axis
    # plt.xticks([])

    plt.yticks(fontsize = tickLabelSize)

    # Set y-limits for modalities
    if modality == 'VASO':
        ticks = np.arange(0, 7, 1)
        plt.yticks(ticks, fontsize=tickLabelSize)
    if modality == 'BOLD':
        ticks = np.arange(0, 19, 3)
        plt.yticks(ticks, fontsize=tickLabelSize)

    plt.legend(loc='upper left', fontsize=legendTextSize)

    plt.savefig(f'../results/Group_v1_{modality}_blocksVsEvents.png', bbox_inches='tight')

    plt.show()

##########################################
########## Plot efficiency data ##########
##########################################

data = pd.read_csv('/Users/sebastiandresbach/git/eventRelatedVASO/results/efficiencyData.csv')
data = data.loc[(data['focus']=='v1')]

palette = {'BOLD': 'tab:orange',
           'VASO': 'tab:blue'
           }

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
ax.set_ylabel('Error', fontsize=labelSize)
ax.set_xlabel('Averaged trials', fontsize=labelSize)

ax.yaxis.set_tick_params(labelsize=tickLabelSize)
ax.xaxis.set_tick_params(labelsize=tickLabelSize)

ax.set_ylim(0, 25)

plt.legend(loc='upper right', fontsize=legendTextSize)

ax.vlines(20, ymin=0, ymax=30, label='5% error', linestyle='dashed', color='white')

# fig.tight_layout()
plt.savefig(f'../results/Group_v1_stabilizing.png', bbox_inches = 'tight')

plt.show()

#########################################
########## Plot efficiency gif ##########
#########################################

data = np.load('/Users/sebastiandresbach/git/eventRelatedVASO/data/trialWiseResponses.npy', allow_pickle=True).item()

from random import seed
from random import choice
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


tmpBOLD = data.loc[(data['nrTrials']==1)& (data['subject']=='sub-07')& (data['run'].str.contains('run-001')) & (data['modality'] == 'BOLD')]
plt.plot(tmpBOLD['currentAverage'].to_numpy())
type(tmpBOLD['currentAverage'].to_numpy()[0])

folder = '/Users/sebastiandresbach/git/eventRelatedVASO/results/tmp'
os.system(f'mkdir {folder}')

seed(5)
palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'
}
for sub in ['sub-07']:
    print(sub)

    # make list of all trials (some where excluded ue to motion)
    subTrials = []
    for key, value in data['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['BOLD']['visiotactile']['layer 1'].items():
        subTrials.append(key)

    # initiate list to dump trials that were already included
    includedTrials = []
    # choose 40 random trials
    for n in range(40):
        # make a figure for each number of trials
        fig = plt.figure()

        # choose a random trial
        selection = choice(subTrials)
        # remove that trial from the list of possible trials
        subTrials.remove(selection)
        # add trialname to list of trials that were already included
        includedTrials.append(selection)

        # get VASO and BOLD responses of first included trial
        tmpBOLD = data['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['BOLD']['visiotactile']['layer 1'][includedTrials[0]]
        tmpVASO = data['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['VASO']['visiotactile']['layer 1'][includedTrials[0]]

        for layer in range(2,4):
            tmpBOLD = np.vstack((tmpBOLD, data['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['BOLD']['visiotactile'][f'layer {layer}'][includedTrials[0]]))
            tmpVASO = np.vstack((tmpVASO, data['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['VASO']['visiotactile'][f'layer {layer}'][includedTrials[0]]))

        for trial in range(0,len(includedTrials)):
            for layer in range(2,4):
                tmpBOLD = np.vstack((tmpBOLD, data['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['BOLD']['visiotactile'][f'layer {layer}'][includedTrials[trial]]))
                tmpVASO = np.vstack((tmpVASO, data['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['VASO']['visiotactile'][f'layer {layer}'][includedTrials[trial]]))

        tmpVASO = np.mean(tmpVASO, axis=0)
        tmpBOLD = np.mean(tmpBOLD, axis=0)

        plt.plot(tmpBOLD, label='_nolegend_', color='tab:orange')
        plt.plot(-tmpVASO, label='_nolegend_', color='tab:blue')

        plt.ylabel('% signal change', fontsize=24)
        plt.xlabel('Time (s)', fontsize=24)


        ticks = range(0,14)
        labels = (np.arange(-4,10)*1.3).round(decimals=1)

        plt.xticks(ticks,labels,rotation=45)


        sns.lineplot(data=layerEventData, x='x', y='data', hue='modality', alpha=0.3, ci=None,palette=palette)

        plt.axvspan(4, 4+(2/1.3), color='grey', alpha=0.2, lw=0, label = 'stimulation')

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
#         plt.xlabel('TR')
        plt.ylim(-2,2.5)
        plt.title(f"Average of {n+1} trials", fontsize=24, pad=20)
        plt.legend(loc='upper left',fontsize=12)
        plt.rcParams['savefig.facecolor']='black'

        plt.savefig(f'{folder}/{sub}_eventRelatedAveragesOf{str(n+1).zfill(2)}Trials.png', bbox_inches='tight')
        plt.show()

import glob
# make gif image
import imageio
images = []
imgs = sorted(glob.glob(f'{folder}/{sub}_eventRelatedAveragesOf*'))

for file in imgs:
    images.append(imageio.imread(file))
imageio.mimsave(f'{folder}/movie.gif', images, duration=0.5)


runData = pd.read_csv('/Users/sebastiandresbach/git/eventRelatedVASO/results/efficiencyData.csv')


subList = []
dataList = []
xList = []
modalityList = []
trialList = []
runList = []
layerList = []
conditionList = []

layers = {'1':'deep','2':'middle','3':'superficial'}

subs = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']
modalities = ['BOLD', 'VASO']

for focus, cmap in zip(['v1'],['tab10']):
    for sub in subs:
        print(sub)

        subRuns = tmp.loc[tmp['subject']==sub]
        runs = sorted(subRuns['run'].unique())

        # runs = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-event*_run-00*_cbv.nii.gz'))

        for run in runs:

            base = os.path.basename(run).rsplit('.', 2)[0]
            print(base)


            for modality in modalities:
                print(modality)

                if 'Random' in base:
                    for trialType in ['visual', 'visiotactile']:


                        for j in range(1, 4):

                            subTrials = []
                            for key, value in data[focus][sub][base][modality][trialType][f'layer {j}'].items():
                                subTrials.append(key)

                            for trial in subTrials[:-1]:

                                for n in range(len(data[focus][sub][base][modality][trialType][f'layer {j}'][trial][0])):

                                    if modality == "BOLD":
                                        dataList.append(data[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

                                    if modality == "VASO":
                                        dataList.append(-data[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

                                    modalityList.append(modality)
                                    trialList.append(trial)
                                    runList.append(base)
                                    xList.append(n)
                                    subList.append(sub)
                                    layerList.append(layers[str(j)])
                                    conditionList.append(trialType)
                else:
                    trialType = 'visiotactile'


                    for j in range(1, 4):

                        subTrials = []
                        for key, value in data[focus][sub][base][modality][trialType][f'layer {j}'].items():
                            subTrials.append(key)

                        for trial in subTrials[:-1]:

                            for n in range(len(data[focus][sub][base][modality][trialType][f'layer {j}'][trial][0])):

                                if modality == "BOLD":
                                    dataList.append(data[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

                                if modality == "VASO":
                                    dataList.append(-data[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

                                modalityList.append(modality)
                                trialList.append(trial)
                                runList.append(base)
                                xList.append(n)
                                subList.append(sub)
                                layerList.append(layers[str(j)])
                                conditionList.append(trialType)


layerEventData = pd.DataFrame({'subject': subList,'x':xList, 'data': dataList, 'modality': modalityList, 'trial': trialList, 'run':runList, 'layer':layerList, 'condition':conditionList})

layerEventData.to_csv('/Users/sebastiandresbach/git/eventRelatedVASO/results/layerEventData.csv')







############################################
########## Plot event FIRs random ##########
############################################

data = pd.read_csv('../../results/backupOld/FIRdataRandom.csv')

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

data = pd.read_csv('../../results/backupOld/blockTimecourses.csv')

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
zscores = pd.read_csv('../../results/backupOld/blocksVsEventsData.csv')
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

data = pd.read_csv('../../results/backupOld/FIR_results.csv')
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
zscores = pd.read_csv('../../results/backupOld/blocksVsEventsData.csv')
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

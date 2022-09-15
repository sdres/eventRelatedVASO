import pandas as pd
import numpy as np
import re



folder = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/sub-05/ses-001/events/'

infile = folder + 'sub-05_ses-001_task-eventStim_run-001.log'
with open(infile) as f:
    f = f.readlines()

triggerTimes = []
for line in f[1:]:
    if re.findall("Keypress: 5",line):
        #print(re.findall("\d+\.\d+", line))
        triggerTimes.append(float(re.findall("\d+\.\d+", line)[0]))


triggerTimes[0] = 0


triggersSubtracted = []
for n in range(len(triggerTimes)-1):
    triggersSubtracted.append(float(triggerTimes[n+1])-float(triggerTimes[n]))

meanFirstTriggerDur = np.mean(triggersSubtracted[::2])
meanSecondTriggerDur = np.mean(triggersSubtracted[1::2])

# find mean trigger-time
meanTriggerDur = (meanFirstTriggerDur+meanSecondTriggerDur)/2

meanTriggerDur


meanFirstTriggerDur
meanSecondTriggerDur




restTimes = []
for line in f[1:]:
    if re.findall("rest",line):
        #print(re.findall("\d+\.\d+", line))
        restTimes.append(float(re.findall("\d+\.\d+", line)[0]))

stimulationTimes = []
for line in f[1:]:
    if re.findall("stimulation",line):
        #print(re.findall("\d+\.\d+", line))
        stimulationTimes.append(float(re.findall("\d+\.\d+", line)[0]))
stimulationTimes
restTimes[0]

realFirstTriggerTime = triggerTimes[0]-meanSecondTriggerDur
realFirstTriggerTime


stimulationTimes[0]-firstTriggerTime
triggerTimes[0]
stimulationTimes[0]
firstTriggerTime
stimulationTimes

restTimes[0]+triggersSubtracted[1]
stimulationTimes[0]+triggersSubtracted[1]
restTimes[1]+triggersSubtracted[1]

restTimes[0]


stimDurations = []
for restTime, stimulationTime in zip(restTimes[1:],stimulationTimes[:-1]):
    stimDurations.append(float(restTime-float(stimulationTime)))
meanStimDuration = np.mean(stimDurations)
actualStimDuration = 2.
discrepancy = meanStimDuration-actualStimDuration
discrepancy
stimDurations

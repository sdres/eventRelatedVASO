# Import libraries
import pandas as pd
import math
from scipy.io import wavfile
import wave
import contextlib
from psychopy import sound, core, prefs, logging, event, visual, gui
from psychopy.sound import Sound
from psychopy.hardware import emulator, keyboard
import numpy as np
import os
import sounddevice as sd
import soundfile as sf
import time
from random import choice

# Load a keyboard in enable abortion.
defaultKeyboard = keyboard.Keyboard()

#***************************
#---EXPERIMENT SETTINGS
#***************************

expName = 'VASO_blockDesignFiveFingers'
expInfo = {'participant': 'subxx',
           'session': '001',
           'run': 1}

# Load a GUI in which the preset parameters can be changed.
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
     core.quit()  # Abort if user pressed cancel


#***************************
#---PREPARE LOGFILE
#***************************

# Define a name so the log-file so it can be attributed to the
# subject/session/run.
# logFilename = f'{expInfo['participant']}_sess{expInfo['session']}_30sOnOff5Fingers_run{expInfo['run']}'
logFileName = '%s_sess%s_30sOnOff_run%s'%(expInfo['participant'], expInfo['session'],expInfo['run'])


## Actually create the log-file.
#logFile = logging.LogFile(logFileName+'.log', level=logging.EXP)
## Make sure we output to the screen, not a file.
#logging.console.setLevel(logging.WARNING)

# save a log file and set level for msg to be received
logFile = logging.LogFile(logFileName+'.log', level=logging.INFO)
logging.console.setLevel(logging.WARNING)  # set console to receive warnings


dateNow = time.strftime("%Y-%m-%d_%H.%M.%S")
logFile.write(f'###############################################\nTHIS EXPERIMENT WAS STARTET {dateNow}\n###############################################\n')



# ****************************************
#-----INITIALIZE GENERIC COMPONENTS
# ****************************************


# Setup the Window
win = visual.Window(
    size=[1920, 1200], fullscr=True, screen=0,
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True,
    units='height')


# Initialize text
instructionsText = 'You will feel your fingers being stimulated. Please lie as still as possible and focus on the stimulation.'
msg = visual.TextStim(win, text=instructionsText, color=(1,1,1), height=50,units='pix',)

fixationCross = visual.TextStim(win=win,
                                text='+',
                                color='white',
                                name='fixationCross')


################### ################### ################
################### Initialize Sound ###################
################### ################### ################

# Set the name/location of the soundfile
soundfile = '30_5_unpredictable.wav'

# Load the sound-file as array, which will act as stimulation "rhythm".
fs, data = wavfile.read(soundfile)

# Because the sound is loaded as an array, we do not know the "real world"
# duration. Therefore, we load it here to calculate it from the number of
# timepoints and their rate.
with contextlib.closing(wave.open(soundfile,'r')) as f:
    frames = f.getnframes()
    sample_rate = f.getframerate()
    duration = frames / float(sample_rate)


soundArr = np.zeros((8, data.shape[0]))
restArr = np.zeros((8, data.shape[0]))

for n in range(0,8):
    soundArr[n] = data

soundArr = np.transpose(soundArr)
restArr = np.transpose(restArr)

soundArr = 2.*(soundArr - np.min(soundArr))/np.ptp(soundArr)-1

# restStim = sound.Sound(value=restArr, secs=30)
# soundStim = sound.Sound(value=soundArr, secs=30)


nTR= 0; # total TR counter
nTR1 = 0; # even TR counter = BOLD
nTR2 = 0; # odd TR counter = VASO


################### ################### ################
################### Initialize Visual ##################
################### ################### ################

tapImages = []
for i in range(0,2):
    tapImages.append(visual.ImageStim(win, pos=[0,0], name='Movie Frame %d'%i,image='visual_%d.png'%(i), units='pix'))

################### ################### ################
################## prepare randomization ################
################### ################### ################

condList = ['visual', 'visiotactile']
visualBlocks = 0
visiotactileBlocks = 0


#########################################################
################## Start of Experiment ##################
#########################################################

globalTime = core.Clock()
fmriTime = core.Clock()
logging.setDefaultClock(fmriTime)
globalTime.reset()
fmriTime.reset()
trialTime = core.Clock()
visStimTime = core.Clock()

msg.draw()
win.flip()


# Waiting for scanner
# Because we want to start odd and even runs differently, we have to wait for the first and second triggers, respectively.
if expInfo['run'] % 2 != 0:
    event.waitKeys(keyList=["5"], timeStamped=False)
    fmriTime.reset()
    nTR= nTR+1; # total TR counter
    nTR1 = nTR1+1; # even TR counter = VASO
    nTR2 = 0; # uneven TR counter = BOLD
    logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))

    logging.data('StartOfRun' + str(expInfo['run']))
    logging.data('StartOfParadigm')

if expInfo['run'] % 2 == 0:
    event.waitKeys(keyList=["5"], timeStamped=False)
    fmriTime.reset()
    nTR= nTR+1; # total TR counter
    nTR1 = nTR1+1; # even TR counter = VASO
    nTR2 = 0; # uneven TR counter = BOLD
    logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))

    logging.data('StartOfRun' + str(expInfo['run']))
    # We are waiting for a second trigger here
    event.waitKeys(keyList=["5"])
    nTR= nTR+1; # total TR counter
    nTR1 = nTR1; # even TR counter = VASO
    nTR2 = nTR2+1; # uneven TR counter = BOLD
    logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))
    logging.data('StartOfParadigm')


fixationCross.draw()
win.flip()


stimDur = 30
restDur = 30
nrTrials = 12
trialCount = 1


# Start with a rest period
logFile.write('\n')
logging.data('rest' + '\n')
trialTime.reset()
while restDur > trialTime.getTime():
    # handle key presses each frame
    for keys in event.getKeys():
        if keys[0] in ['escape', 'q']:
            myWin.close()
            core.quit()
        elif keys in ['5']:
            nTR = nTR + 1
            if nTR % 2 ==1: # odd TRs
                nTR1 = nTR1 + 1

            elif nTR % 2 == 0:
                nTR2 = nTR2 +1

            logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))

visStimType = 0
# Start with trials
while trialCount <= nrTrials:
    trialTime.reset()
    # make sure that not more than 6 blocks are of one condition
    if visiotactileBlocks == 6:
        condList = ['visual']
    if visualBlocks == 6:
        condList = ['visiotactile']
    # randomly choose condition
    tmp_choice = choice(condList)

    logFile.write('\n')
    if tmp_choice == 'visiotactile':
        visiotactileBlocks = visiotactileBlocks +1
        logging.data('visiotactile stimulation' + '\n')
        sd.play(soundArr, fs, device = 23)
    else:
        logging.data('visual stimulation' + '\n')
        visualBlocks = visualBlocks +1
    logging.data('stimulation started')

    while stimDur > trialTime.getTime():

        if visStimTime.getTime() >= 1/16:
            if visStimType == 0:
                tapImages[visStimType].draw()
                win.flip()
                visStimTime.reset()
                visStimType = 1
        if visStimTime.getTime() >= 1/16:
            if visStimType == 1:
                tapImages[visStimType].draw()
                win.flip()
                visStimTime.reset()
                visStimType = 0

        # handle key presses each frame
        for keys in event.getKeys():
            if keys[0] in ['escape', 'q']:
                myWin.close()
                core.quit()
            elif keys in ['5']:
                nTR = nTR + 1
                if nTR % 2 ==1: # odd TRs
                    nTR1 = nTR1 + 1

                elif nTR % 2 == 0:
                    nTR2 = nTR2 +1

                logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))

    if tmp_choice == 'visiotactile':
        sd.stop()
    logging.data('stimulation stopped' + '\n')
    fixationCross.draw()
    win.flip()
    visStimCount = 0

    trialTime.reset()
    logging.data('rest' + '\n')
    while restDur > trialTime.getTime():
        # handle key presses each frame
        for keys in event.getKeys():
            if keys[0] in ['escape', 'q']:
                myWin.close()
                core.quit()
            elif keys in ['t']:
                nTR = nTR + 1
                if nTR % 2 ==1: # odd TRs
                    nTR1 = nTR1 + 1

                elif nTR % 2 == 0:
                    nTR2 = nTR2 +1

                logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))

    trialCount = trialCount + 1


logging.data('EndOfRun' + str(expInfo['run']) + '\n')

win.close()
core.quit()

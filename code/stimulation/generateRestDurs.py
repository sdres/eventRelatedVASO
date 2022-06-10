
'''

To do:

    - make sure that old designs are not automatically overwritten


'''
from psychopy import gui,data
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


psychopyVersion = '2021.1.2'
expName = 'generateDesign'


expInfo = {'participant': 'subxx', 'session': '001', 'inter trial interval min (ms)': 3000, 'inter trial interval max (ms)': 10000, 'run duration (s)': 750}


# dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
# if dlg.OK == False:
#     core.quit()  # user pressed cancel
# expInfo['date'] = data.getDateStr()  # add a simple timestamp
# expInfo['expName'] = expName
# expInfo['psychopyVersion'] = psychopyVersion

firstRest = 10000 + np.random.uniform(expInfo['inter trial interval min (ms)'],expInfo['inter trial interval max (ms)'])


# initiate a list and start with a minimum first rest period.
restTimesList = [firstRest]



lastPossibleTrial = expInfo['run duration (s)']*1000 - 10000 # Make sure that there is some rest at the end.


timeLeft = lastPossibleTrial

time = firstRest


while time < timeLeft:
    randDuration = np.random.uniform(expInfo['inter trial interval min (ms)'],expInfo['inter trial interval max (ms)'])

    restTimesList.append(randDuration)

    time = time + randDuration + 2000


restTimesArr = np.asarray(restTimesList)
restTimesArr = (restTimesArr/1000).round(3)
plt.hist(restTimesArr[1:])

np.savetxt("eventRelatedTemplate.csv", restTimesArr, delimiter= " ", fmt = '%1.3f')


# Length test
# Add stimulation time to each rest period
lengthTest = restTimesArr+2

sum = 0;

#Loop through the array to calculate sum of elements
for i in range(0, len(lengthTest)):
   sum = sum + lengthTest[i];

print("Sum of all the elements of an array: " + str(sum));

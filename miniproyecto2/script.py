import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_tissue_data, read_tissue_data
from dipy.segment.tissue import TissueClassifierHMRF
import nibabel as nib
import time
from util import *
import time

import os

lista = [x[2] for x in os.walk('./datos/')]
lista2 = [x[0] for x in os.walk('./datos/')]
patient = 1
for i,(l) in enumerate(lista):
    if len(l) != 0:
        print '::::::::::::::::::::::::::::::::::'
        print ':::::: reading patient n',patient,':::::::'
        print '::::::::::::::::::::::::::::::::::'
        dir1 = lista2[i]+'/'+l[0]
        dir2 = lista2[i]+'/'+l[2]
        img = nib.load(dir1)
        img2 = nib.load(dir2)
        t1 = img.get_data()
        t2 = img2.get_data()

        #hyperparameters
        nclass = 3
        beta = 0.1

        #fitting the model
        t0 = time.time()

        hmrf = TissueClassifierHMRF()
        initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass, beta)
        
        t1 = time.time()
        total_time = t1-t0
        print('Total time:' + str(total_time))

        #merging brain hemispheres
        dice_max = 0
        dice_max_l = 0
        dice_max_r = 0
        matches = []
        for i in range(0,final_segmentation.shape[2]):
            #getting whole brain slide
            wholenewbrain, whole_brain = get_whole_brain(i, t2, final_segmentation)
            #matching
            dice, A_B = get_score(wholenewbrain, whole_brain)
            matches.append(A_B)
            if dice > dice_max:
                dice_max = dice
                model = wholenewbrain
                real = whole_brain
                n_slide = i

        #calculating partial scores
        lef, rig = get_hemispheres(n_slide,t2)

        model_l = np.where(model==lef, 1, 0)*lef
        model_r = np.where(model==rig, 1, 0)*rig
        model_l = model_l[model_l!=0]
        model_r = model_r[model_r != 0]

        lef_ = lef[lef != 0]
        rig_ = rig[rig != 0]

        dice2 = (model_l.shape[0]*2.0 )/ (lef_.shape[0] + model_l.shape[0])
        dice3 = (model_r.shape[0]*2.0 )/ (rig_.shape[0] + model_r.shape[0])

        print 'Sorensen-Dice for whole brain: ',dice_max
        print 'Sorensen-Dice for left side: ',dice2
        print 'Sorensen-Dice for right side: ',dice3

        name = str(patient)+'.txt'
        print 'name: ',name
        f=open(name,"w")
        max1 = 'Sorensen-Dice for whole brain: '+ str(dice_max)
        max2 = 'Sorensen-Dice for left side: '+str(dice2)
        max3 = 'Sorensen-Dice for right side: '+str(dice3)
        print >> f ,max1
        print >> f ,max2
        print >> f ,max3

        patient+=1

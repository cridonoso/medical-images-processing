import numpy as np

def get_whole_brain(index, t2, final_segmentation):
    slide2 = t2[...,index]
    #we select both hemispheres (and its respective tissue) by using database code //gray = 1 white = 2 ioc = 0
    left_gray = np.where(slide2>999, 2, 0) & np.where(slide2<1035, 2, 0)
    right_gray = np.where(slide2>1999, 2, 0) & np.where(slide2<2035, 2, 0)

    left_white = np.where(slide2 == 2, 1, 0)
    right_white = np.where(slide2 == 41, 1, 0)

    #we merge everything making whole brain
    gray = np.rot90(right_gray + left_gray)
    white = np.rot90(right_white + left_white)
    whole_brain = gray + white

    slide_ = np.rot90(final_segmentation[...,index])
    gray_matter = np.where(slide_ == 2, 2, 0)
    white_matter = np.where(slide_ == 3, 1, 0)
    wholenewbrain = gray_matter + white_matter

    return wholenewbrain, whole_brain

def get_score(wholenewbrain,whole_brain):
    #matching
    dice = 0
    part = wholenewbrain[whole_brain==wholenewbrain]
    part = part[part!=0]
    A_B = part.shape[0]
    A = wholenewbrain[wholenewbrain != 0].shape[0]
    B = whole_brain[whole_brain != 0].shape[0]
    if A != 0 or B != 0:
        dice = A_B*2.0 / (A + B)
    return dice, A_B

def get_hemispheres(index, t2):
    slide2 = t2[...,index]
    #we select both hemispheres (and its respective tissue) by using database code //gray = 1 white = 2 ioc = 0
    left_gray = np.where(slide2>999, 2, 0) & np.where(slide2<1035, 2, 0)
    right_gray = np.where(slide2>1999, 2, 0) & np.where(slide2<2035, 2, 0)

    left_white = np.where(slide2 == 2, 1, 0)
    right_white = np.where(slide2 == 41, 1, 0)

    right = left_gray+left_white
    left  = right_gray+right_white

    return np.rot90(left), np.rot90(right)

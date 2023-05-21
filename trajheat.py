import os
import numpy as np
import matplotlib.pyplot as plt


path = r'J:\map\latex\期刊扩展\fig\trajheat\all_new1.txt'
#path = r'J:\map\latex\期刊扩展\fig\trajheat\all_new.txt'
#           0       1       2           3       4       5       6     7     8       9
names = ['coord', 'GT', 'RPE_Full', 'RPE_R', 'RPE_T', 'PNE', 'CPV', 'MME', 'MPV', 'MOM']

def heatScatter(data, col, m):
    '''
    默认使用data第1，2列作为xy
    col指定对应列用于配置颜色
    '''
    x = data[:, 0]
    y = data[:, 1]
    color = data[:, col]

    f, ax = plt.subplots()
    plt.xlim(-220, 80)
    plt.ylim(-31, 19)

    ax.set_box_aspect(1/6)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    print(names[col-2])
    ax.set_title(names[col-2])
    points = ax.scatter(x, y, c=color, s=10, cmap="rainbow", vmin=0, vmax=1)
    f.colorbar(points)
    plt.savefig(os.path.join(r"J:\map\latex\期刊扩展\fig\trajheat", names[col-2] + '.pdf'), dpi=100)
    # plt.show()
    plt.close()

def normalized(list):
    #norm = np.linalg.norm(list)
    mmax = np.max(list)
    mmin = np.min(list)
    arr = (list - mmin) / (mmax - mmin)
    return arr

def readTxt(path):
    coord = []
    GT = []
    RPE_Full = []
    RPE_R = []
    RPE_T = []
    PNE = []
    CPV = []
    MME = []
    MPV = []
    MOM = []
    with open(path, encoding='utf-8') as file:
        logs = file.readlines()
    for i, l in enumerate(logs):
        if "x-y-z" in l:  # coord
            coord = np.asarray(l.replace('x-y-z ', '').split(' '), dtype=np.float64)
            #print(len(coord) / 3)
        if "GT" in l:  # GT noise
            GT = np.asarray(l.replace('GT ', '').split(' '), dtype=np.float64)
            #print(len(GT))
        if "RPE-Full" in l:
            RPE_Full = np.asarray(l.replace('RPE-Full ', '').split(' '), dtype=np.float64)
            #print(len(RPE_Full))
        if "RPE-T" in l:
            RPE_T = np.asarray(l.replace('RPE-T ', '').split(' '), dtype=np.float64)
            #print(len(RPE_T))
        if "RPE-R" in l:
            RPE_R = np.asarray(l.replace('RPE-R ', '').split(' '), dtype=np.float64)
            #print(len(RPE_R))
        if "PNE-est" in l:
            #aaa = l.replace('PNE-est ', '').split(' ')
            PNE = np.asarray(l.replace('PNE-est ', '').split(' '), dtype=np.float64)
            #print(len(PNE))
        if "CPV-est" in l:
            CPV = np.asarray(l.replace('CPV-est ', '').split(' '), dtype=np.float64)
            #print(len(CPV))
        if "MME-est" in l:
            MME = np.asarray(l.replace('MME-est ', '').split(' '), dtype=np.float64)
            #print(len(CPV))
        if "MPV-est" in l:
            MPV = np.asarray(l.replace('MPV-est ', '').split(' '), dtype=np.float64)
            #print(len(CPV))
        if "MOM-est" in l:
            MOM = np.asarray(l.replace('MOM-est ', '').split(' '), dtype=np.float64)
            #print(len(CPV))


    coord = coord.reshape(-1, 3)  # 0-1-2

    GT = GT.reshape(-1, 1)  # 3
    RPE_Full = RPE_Full.reshape(-1, 1)  # 4
    RPE_R = RPE_R.reshape(-1, 1)  # 5
    RPE_T = RPE_T.reshape(-1, 1)  # 6
    PNE = PNE.reshape(-1, 1)  # 7
    CPV = CPV.reshape(-1, 1)  # 8

    MME = MME.reshape(-1, 1)  # 9
    MPV = MPV.reshape(-1, 1)  # 10
    MOM = MOM.reshape(-1, 1)  # 11


    GT = normalized(GT)

    RPE_Full = normalized(RPE_Full)
    RPE_R = normalized(RPE_R)
    RPE_T = normalized(RPE_T)

    PNE = normalized(PNE)
    CPV = normalized(CPV)

    MME = normalized(MME)
    MPV = normalized(MPV)
    MOM = normalized(MOM)
    #                0    1     2           3       4       5       6       7       8       9
    # names = ['coord', 'GT', 'RPE_Full', 'RPE_R', 'RPE_T', 'PNE', 'CPV', 'MME', 'MPV', 'MOM']
    new_data = np.hstack((coord, GT, RPE_Full, RPE_R, RPE_T, PNE, CPV, MME, MPV, MOM))
    np.savetxt(r"J:\map\latex\期刊扩展\fig\trajheat\1111.txt", new_data, fmt='%.3f')
    #heatScatter(new_data, 3, 'GT')
    heatScatter(new_data, 4, 'RPE_Full')
    heatScatter(new_data, 5, 'RPE_R')
    heatScatter(new_data, 6, 'RPE_T')
    heatScatter(new_data, 7, 'PNE')
    heatScatter(new_data, 8, 'CPV')
    #heatScatter(new_data, 9, 'MME')
    #heatScatter(new_data, 10, 'MPV')
    #heatScatter(new_data, 11, 'MOM')


readTxt(path)
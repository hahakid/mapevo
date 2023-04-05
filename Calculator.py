import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from correlations import multirelation, multiple_correlation_coefficient, adjusted_multiple_correlation_coefficient

import matplotlib.pyplot as plt

'''
计算各个指标之间得 二元相关性 和 三元
每个log
1. 时间复杂度
2. window
3. noise
4. map-based metric
    mme
    mpv
    mom
5. reference-based metric
    rpe-full
    rpe-t
    rpe-r
6. new-plane-based metrics:
    Plane-MOM
    PNE
    PPV
'''

def BinaryCorrelation(a, b):
    cor_pearsonr = abs(pearsonr(a, b).statistic)
    cor_spearmanr = abs(spearmanr(a, b).statistic)
    cor_kendalltau = abs(kendalltau(a, b).statistic)
    return [cor_pearsonr, cor_spearmanr, cor_kendalltau]

def TernaryCorrelation(a, b, c):
    T1 = multirelation(a, b, c)
    T2 = multiple_correlation_coefficient(a, b, c) #, metric=pearsonr)
    T3 = adjusted_multiple_correlation_coefficient(a, b, c) #, metric=pearsonr)
    return [T1, T2, T3]

def readTxt(path, mode):
    '''
    :param path:
    :return:
    1. noise
    2. map-based metrics:
        2.1 mme
        2.2 mpv
        2.3 mom
    3. reference-based metrics:
        3.1 rt
        3.2 t
        3.3 r
    '''

    # logs = []
    mmeTime = []
    mpvTime = []
    momTime = []
    spvTime = []
    pneTime = []
    cpvTime = []

    GT_Noise = []
    MME = []
    MPV = []
    MOM = []
    RPE_Full = []
    RPE_T = []
    RPE_R = []
    Plane_MOM = []
    PNE = []
    CPV = []
    CPN = []

    with open(path, encoding='utf-8') as file:
        logs = file.readlines()
        #logs.append(content)
    # print(logs)
    for i, l in enumerate(logs):
        #print(i, l)
        if "GT Noise" in l:
            GT_Noise = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        if "MME" in l:
            MME = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)
            #print(MME)

        if "MPV" in l:
            MPV = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)
            #print(MME)

        if "MOM" in l:
            MOM = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)
            #print(MME)

        if "RPE-Full" in l:
            RPE_Full = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)
            #print(MME)

        if "RPE-T" in l:
            RPE_T = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)
            #print(MME)

        if "RPE-R" in l:
            aaa = l.split(']')[0].split('[')[1].split(',')
            RPE_R = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        #if "Plane-MOM" in l:
        if "PMO" in l:
            Plane_MOM = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        if "PNE: [" in l:
            #print(l)
            PNE = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        if "CPV: [" in l:
            CPV = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        if "CPN: [" in l:
            CPN = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        # 最后处理时间因为要跳越
        if "time consume" in l:
            #print(logs[i])
            mmeTime.append(float(logs[i+1].split(' ')[2]))
            mpvTime.append(float(logs[i+2].split(' ')[2]))
            momTime.append(float(logs[i+3].split(' ')[2]))
            spvTime.append(float(logs[i+4].split(' ')[2]))
            pneTime.append(float(logs[i+5].split(' ')[2]))
            cpvTime.append(float(logs[i+6].split(' ')[2]))
            i += 6

    mme_avg_time = np.mean(np.asarray(mmeTime))
    mpv_avg_time = np.mean(np.asarray(mpvTime))
    mom_avg_time = np.mean(np.asarray(momTime))
    spv_avg_time = np.mean(np.asarray(spvTime))
    pne_avg_time = np.mean(np.asarray(pneTime))
    cpv_avg_time = np.mean(np.asarray(cpvTime))
    time_consumption = [mme_avg_time, mpv_avg_time, mom_avg_time, spv_avg_time, pne_avg_time, cpv_avg_time]

    if mode == 't':
        #data1 = GT_Noise
        data1 = RPE_T
    if mode == 'r':
        #data1 = GT_Noise
        data1 = RPE_R
    if mode == 'rt':
        #data1 = GT_Noise
        data1 = RPE_Full
    # data1 = GT_Noise  # 开启后，直接通过GT_Noise 计算二元相关性，三元相关性则失效
    # print(len(GT_Noise), len(data1), len(MME), len(MPV), len(MOM), len(Plane_MOM), len(PNE), len(PPV))
    # data1 = GT_Noise
    #assert len(GT_Noise) == len(data1) == len(MME) == len(MPV) == len(MOM) == len(Plane_MOM) == len(PNE) == len(PPV) == len(MPV)
    corrList1 = []
    for data2 in [MME, MPV, MOM, Plane_MOM, PNE, CPV, CPN]:
        [c_pearsonr, c_spearmanr, c_kendalltau] = BinaryCorrelation(data1, data2)
        # [tc1, tc2, tc3] = TernaryCorrelation(GT_Noise, data2, data1)
        corrList1.append([c_pearsonr, c_spearmanr, c_kendalltau])#, tc1, tc2, tc3])

    corrList2 = []
    if mode == 'rt':
        [tc1, tc2, tc3] = TernaryCorrelation(data1, Plane_MOM, PNE)
        [tc4, tc5, tc6] = TernaryCorrelation(data1, CPV, CPN)
        corrList2.append([tc1, tc2, tc3])
        corrList2.append([tc4, tc5, tc6])

    return time_consumption, corrList1, corrList2


def readTxt1(path, mode):
    '''
    :param path:
    :return:
    1. noise
    2. map-based metrics:
        2.1 mme
        2.2 mpv
        2.3 mom
    3. reference-based metrics:
        3.1 rt
        3.2 t
        3.3 r
    '''

    # logs = []
    mmeTime = []
    mpvTime = []
    momTime = []
    spvTime = []
    pneTime = []
    cpvTime = []

    GT_Noise = []
    MME = []
    MPV = []
    MOM = []
    RPE_Full = []
    RPE_T = []
    RPE_R = []
    Plane_MOM = []
    PNE = []
    CPV = []
    CPN = []

    with open(path, encoding='utf-8') as file:
        logs = file.readlines()
        #logs.append(content)
    # print(logs)
    for i, l in enumerate(logs):
        #print(i, l)
        if "GT Noise: [" in l:
            GT_Noise = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        #if "MME" in l:
        #    MME = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)
            #print(MME)

        #if "MPV" in l:
        #    MPV = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)
            #print(MME)

        #if "MOM" in l:
        #    MOM = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)
            #print(MME)

        if "RPE-Full" in l:
            RPE_Full = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)
            #print(MME)

        if "RPE-T" in l:
            RPE_T = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)
            #print(MME)

        if "RPE-R" in l:
            aaa = l.split(']')[0].split('[')[1].split(',')
            RPE_R = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        if "PMO: [" in l:
            Plane_MOM = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        if "PNE: [" in l:
            #print(l)
            PNE = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        if "CPV: [" in l:
            CPV = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        if "CPN: [" in l:
            CPN = np.asarray(l.split(']')[0].split('[')[1].split(','), dtype=np.float64)

        # 最后处理时间因为要跳越
        '''
        if "time consume" in l:
            #print(logs[i])
            mmeTime.append(float(logs[i+1].split(' ')[2]))
            mpvTime.append(float(logs[i+2].split(' ')[2]))
            momTime.append(float(logs[i+3].split(' ')[2]))
            spvTime.append(float(logs[i+4].split(' ')[2]))
            pneTime.append(float(logs[i+5].split(' ')[2]))
            cpvTime.append(float(logs[i+6].split(' ')[2]))
            i += 6

    mme_avg_time = np.mean(np.asarray(mmeTime))
    mpv_avg_time = np.mean(np.asarray(mpvTime))
    mom_avg_time = np.mean(np.asarray(momTime))
    spv_avg_time = np.mean(np.asarray(spvTime))
    pne_avg_time = np.mean(np.asarray(pneTime))
    cpv_avg_time = np.mean(np.asarray(cpvTime))
    time_consumption = [mme_avg_time, mpv_avg_time, mom_avg_time, spv_avg_time, pne_avg_time, cpv_avg_time]
        '''
    if mode == 't':
        #data1 = GT_Noise
        data1 = RPE_T
    if mode == 'r':
        #data1 = GT_Noise
        data1 = RPE_R
    if mode == 'rt':
        #data1 = GT_Noise
        data1 = RPE_Full
    # data1 = GT_Noise
    # print(len(GT_Noise), len(data1), len(MME), len(MPV), len(MOM), len(Plane_MOM), len(PNE), len(PPV))
    # data1 = GT_Noise
    #assert len(GT_Noise) == len(data1) == len(MME) == len(MPV) == len(MOM) == len(Plane_MOM) == len(PNE) == len(PPV) == len(MPV)
    corrList = []
    for data2 in [Plane_MOM, PNE, CPV, CPN]:
        [c_pearsonr, c_spearmanr, c_kendalltau] = BinaryCorrelation(data1, data2)
        [tc1, tc2, tc3] = TernaryCorrelation(GT_Noise, data2, data1)
        corrList.append([c_pearsonr, c_spearmanr, c_kendalltau, tc1, tc2, tc3])

    return 0, corrList

def readTime(path):
    with open(path, encoding='utf-8') as file:
        logs = file.readlines()
    mmeTime = []
    mpvTime = []
    momTime = []
    spvTime = []
    pneTime = []
    cpvTime = []
    for i, l in enumerate(logs):
        if "time consume" in l:
            #print(logs[i])
            mmeTime.append(float(logs[i+1].split(' ')[2]))
            mpvTime.append(float(logs[i+2].split(' ')[2]))
            momTime.append(float(logs[i+3].split(' ')[2]))
            spvTime.append(float(logs[i+4].split(' ')[2]))
            pneTime.append(float(logs[i+5].split(' ')[2]))
            cpvTime.append(float(logs[i+6].split(' ')[2]))
            i += 6

    mme_ = np.asarray(mmeTime)
    mpv_ = np.asarray(mpvTime)
    mom_ = np.asarray(momTime)
    spv_ = np.asarray(spvTime)
    pne_ = np.asarray(pneTime)
    cpv_ = np.asarray(cpvTime)
    ttime = np.vstack((mme_, mpv_, mom_, spv_, pne_, cpv_)).T
    np.savetxt(path.replace('.txt', '_time.txt'), ttime, fmt='%.4f')



if __name__ == '__main__':

    #readTime(r'J:\map\code\src\results\时间复杂度分析5-10-15\win=5 mode=t scale=1.5.txt')
    #readTime(r'J:\map\code\src\results\时间复杂度分析5-10-15\win=10 mode=t scale=1.5.txt')
    #readTime(r'J:\map\code\src\results\时间复杂度分析5-10-15\win=15 mode=t scale=1.5.txt')
    #readTime(r'J:\map\latex\fig\timec\357\win=3 mode=rt scale=2.0.txt')
    #readTime(r'J:\map\latex\fig\timec\357\win=7 mode=rt scale=2.0.txt')

    fatherPath = r'J:\map\code\src\results'#\357'
    scenes = ['floor2-1']  #['floor2', 'garage', 'office1', 'office2']
    #arg_list = []
    for winSize in range(5, 16, 5):
    #for winSize in range(3, 8, 2):
        for mode in ['t', 'r', 'rt']:
            for scale in [1, 1.5, 2]:
                #print(winSize, mode, scale)
                FileName = "win=%d mode=%s scale=%.1f.txt" % (winSize, mode, scale)
                titles = FileName.replace('.txt', '')
                FileName = os.path.join(fatherPath, scenes[0], FileName)
                #FileName = os.path.join(fatherPath, FileName)
                outFileName = FileName.replace('.txt', '_cor.txt')
                print(FileName)
                if os.path.exists(FileName):
                    time, corr1, corr2 = readTxt(FileName, mode)
                    corr = corr1
                    #''' write log
                    metricName = ['MME', 'MPV', 'MOM', 'Plane_MOM', 'PNE', 'CPV', 'CPN']
                    with open(outFileName, 'a') as sfile:
                        sfile.writelines("MME MPV MOM Plane_MOM PNE CPV CPN\n")
                        sfile.writelines(str(time)+"\n")
                        sfile.writelines("pearsonr spearmanr kendalltau tc1 tc2 tc3\n")
                        for i in range(len(corr)):
                            sfile.writelines(metricName[i]+" "+str(corr[i]) + "\n")
                        if mode == 'rt':
                            for i in range(len(corr2)):
                                sfile.writelines("tricor"+str(i)+" "+str(corr2[i]) + "\n")
                    #'''
                    #  plot
                    labels = ('Pearson', 'Spearman', 'Kendall') #
                    #labels = ('Pearson', 'Spearman', 'Kendallta', 'MultiRel', 'MCC', 'adjustedMCC') #
                    #labels = ('Pearson', 'Spearman', 'Kendallta', 'MultiRel') #
                    metrics = ['MME', 'MPV', 'MOM', 'Plane_MOM', 'PNE', 'CPV', 'CPN']
                    metricDict = {}  # save as dict = {key=metric, value=tuple(data)}
                    corrDraw = np.asarray(corr, )[:, :len(labels)]
                    corrDraw = np.around(corrDraw, 2)
                    for i in range(corrDraw.shape[0]):
                        metricDict[metrics[i]] = tuple(corrDraw[i])
                    x = np.arange(len(labels))
                    width = 0.125
                    bias = 0
                    fig, axes = plt.subplots()
                    outImgName = outFileName.replace('_cor.txt', '.pdf')
                    hatch = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
                    for key, value in metricDict.items():
                        offset = width * bias
                        rects = axes.bar(x+offset, value, width, label=key, hatch=hatch[bias])
                        axes.bar_label(rects, padding=2)
                        bias += 1
                    axes.legend(loc='upper left', bbox_to_anchor=(0.62, 1))
                    #axes.set_title(titles, fontdict={"family": "Times New Roman", "size": 20})
                    plt.ylabel('Correlation', fontdict={"family": "Times New Roman", "size": 20})
                    plt.xlabel('Metrics', fontdict={"family": "Times New Roman", "size": 20})
                    plt.grid(axis='y', color='0.95')
                    plt.savefig(outImgName, dpi=100)
                    # plt.show()
                    plt.close()


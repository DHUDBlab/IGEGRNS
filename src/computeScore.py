import numpy as np
from itertools import product, permutations, combinations, combinations_with_replacement
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd


# 构建真实调控边字典，用于计算分数
def createTrueDict(trueEdgesDF, geneName, directed=True, selfEdges=True):
    # 设置输出格式为全部输出
    np.set_printoptions(threshold=np.inf)
    top_k = 0
    # 如果有向
    if directed:
        if selfEdges:
            possibleEdges = list(product(geneName, repeat=2))
        else:
            possibleEdges = list(permutations(geneName))

        TrueEdgeDict = {'|'.join(p): 0 for p in possibleEdges}
        # 构建真实调控边字典
        # 1 表示存在调控边
        # 0 不存在调控边
        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[(trueEdgesDF['Gene1'] == key.split('|')[0]) &
                                   (trueEdgesDF['Gene2'] == key.split('|')[1])]) > 0:
                TrueEdgeDict[key] = 1
                top_k += 1
    # 无向
    else:
        if selfEdges:
            possibleEdges = list(combinations_with_replacement(geneName, r=2))
        else:
            possibleEdges = list(combinations(geneName, r=2))
        TrueEdgeDict = {'|'.join(p): 0 for p in possibleEdges}

        # 构建真实调控边字典，1 表示存在调控边，0 不存在调控边
        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[((trueEdgesDF['Gene1'] == key.split('|')[0]) &
                                    (trueEdgesDF['Gene2'] == key.split('|')[1])) |
                                   ((trueEdgesDF['Gene2'] == key.split('|')[0]) &
                                    (trueEdgesDF['Gene1'] == key.split('|')[1]))]) > 0:
                TrueEdgeDict[key] = 1
                top_k += 1

    return TrueEdgeDict, top_k



# 构建预测调控边字典，用于计算分数
def createPredDict(predEdgeDF, geneName, directed=True, selfEdges=True):
    # 如果有向
    if directed:
        if selfEdges:
            possibleEdges = list(product(geneName, repeat=2))
        else:
            possibleEdges = list(permutations(geneName))

        PredEdgeDict = {'|'.join(p): 0 for p in possibleEdges}
        # 构建预测调控边字典
        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[(predEdgeDF['Gene1'] == key.split('|')[0]) &
                                   (predEdgeDF['Gene2'] == key.split('|')[1])]
            if len(subDF) > 0:
                PredEdgeDict[key] = np.abs(subDF["score"].values[0])
                # PredEdgeDict[key] = 1
    # 无向
    else:
        if selfEdges:
            possibleEdges = list(combinations_with_replacement(geneName, r=2))
        else:
            possibleEdges = list(combinations(geneName, r=2))

        PredEdgeDict = {'|'.join(p): 0 for p in possibleEdges}

        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[((predEdgeDF['Gene1'] == key.split('|')[0]) &
                                    (predEdgeDF['Gene2'] == key.split('|')[1])) |
                                   ((predEdgeDF['Gene2'] == key.split('|')[0]) &
                                    (predEdgeDF['Gene1'] == key.split('|')[1]))]
            if len(subDF) > 0:
                PredEdgeDict[key] = max(np.abs(subDF["score"].values))
                # PredEdgeDict[key] = 1
    return PredEdgeDict



def computeScore(TrueEdgeDict, PredEdgeDict):
    """
    :param TrueEdgeDict: 真实调控边字典
    :param PredEdgeDict: 预测调控边字典
    :return: 计算评测分数 AUROC、AUPRC
    """
    outDF = pd.DataFrame([TrueEdgeDict, PredEdgeDict]).T
    outDF.columns = ['TrueEdges', 'PredEdges']
    # print(outDF)
    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)

    return auc(recall, prec), auc(fpr, tpr)

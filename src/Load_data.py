"""
-*- coding: utf-8 -*-
@Author : Smartpig
@Institution : DHU/DBLab
@Time : 2022/3/27 16:45
"""

"""
-*- coding: utf-8 -*-
@Author : Smartpig
@Institution : DHU/DBLab
@Time : 2022/3/27 16:44
"""

import pandas as pd
import os





def gen_geneNameDict(datasetPath, geneExpData, refNetwork):
    # Return gene dictionary about TF and Target
    geneName = pd.DataFrame({'geneName': geneExpData,
                             'index': range(geneExpData.shape[0])})
    geneName.to_csv(datasetPath + 'geneNameDict.csv', index=True, sep=',')
    geneNameDict = dict(zip(geneName['geneName'], geneName['index']))
    tf = refNetwork['Gene1'].drop_duplicates()
    target = refNetwork['Gene2'].drop_duplicates()

    tfDic = []
    for i in range(tf.shape[0]):
        temp = tf.iloc[i]
        tfdic = [temp, geneNameDict[temp]]
        tfDic.append(tfdic)

    targetDic = []
    for i in range(target.shape[0]):
        temp = target.iloc[i]
        targetdic = [temp, geneNameDict[temp]]
        targetDic.append(targetdic)

    edgeDic = []
    for i in range(refNetwork.shape[0]):
        edge = [geneNameDict[refNetwork.iloc[i, 0]], geneNameDict[refNetwork.iloc[i, 1]]]
        edgeDic.append(edge)

    TF = pd.DataFrame(tfDic, columns=['TF', 'index'])
    Target = pd.DataFrame(targetDic, columns=['Target', 'index'])
    Edge = pd.DataFrame(edgeDic, columns=['TF', 'Target'])

    TF.to_csv(datasetPath + 'TF.csv', index=True, sep=',')
    Target.to_csv(datasetPath + 'Target.csv', index=True, sep=',')
    Edge.to_csv(datasetPath + 'Edge.csv', index=True, sep=',')
    return TF, Target, Edge


def gen_dataSplit(datasetPath, TF, Target, Edge, k):
    posEdge = []
    negEdge = []
    for i in range(TF.shape[0]):
        tf = TF.iloc[i, 1]
        for j in range(Target.shape[0]):
            target = Target.iloc[j, 1]
            flag = Edge.loc[(Edge['TF'] == tf) & (Edge['Target'] == target)]
            if flag.empty == False:
                posEdge.append([tf, target, 1])
            else:
                negEdge.append([tf, target, 0])
    posEdge = pd.DataFrame(posEdge, columns=['TF', 'Target', 'Label']).sample(frac=1)
    negEdge = pd.DataFrame(negEdge, columns=['TF', 'Target', 'Label']).sample(frac=1)
    posEdgeNum = posEdge.shape[0]
    negEdgeNum = negEdge.shape[0]

    # train_df = posEdge.iloc[:int(posEdgeNum * trainRatio)].append(negEdge.iloc[:int(negEdgeNum * trainRatio)])
    # val_df = posEdge.iloc[int(posEdgeNum * trainRatio):int(posEdgeNum * (trainRatio + valRatio))].append(
    #     negEdge.iloc[int(negEdgeNum * trainRatio):int(negEdgeNum * (trainRatio + valRatio))])
    # test_df = posEdge.iloc[int(posEdgeNum * (trainRatio + valRatio)):].append(
    #     negEdge.iloc[int(negEdgeNum * (trainRatio + valRatio)):])
    #
    # train_df.to_csv(dataPath + 'train_set.csv', index=True, sep=',')
    # val_df.to_csv(dataPath + 'val_set.csv', index=True, sep=',')
    # test_df.to_csv(dataPath + 'test_set.csv', index=True, sep=',')
    os.chdir(datasetPath)
    splitPath = 'k-fold'
    if not os.path.exists(splitPath):
        os.mkdir(splitPath)
    splitPath = datasetPath+splitPath+'/'

    for i in range(k):
        split_df = posEdge.iloc[int(posEdgeNum*(i/k)):int(posEdgeNum*((i+1)/k))]\
            .append(negEdge.iloc[int(negEdgeNum*(i/k)):int(negEdgeNum*((i+1)/k))])
        split_df.to_csv(splitPath+str(i)+'.csv', index=True, sep=',')
    return posEdgeNum/(posEdgeNum+negEdgeNum)



def gen_split(dataPath, args):
    os.chdir(dataPath)
    # generate work path
    datasetPath = args.datasetPath
    if not os.path.exists(datasetPath):
        os.mkdir(datasetPath)
    datasetPath = dataPath+datasetPath+'/'

    geneExpData = pd.read_csv(args.geneDataName, sep=',', header=0, index_col=None)
    refNetwork = pd.read_csv(args.refNetworkName, sep=',', header=0, index_col=None)

    TF, Target, Edge = gen_geneNameDict(datasetPath, geneExpData.iloc[:, 0], refNetwork.iloc[:, 0:2])
    density = gen_dataSplit(datasetPath, TF, Target, Edge, args.k)
    return density
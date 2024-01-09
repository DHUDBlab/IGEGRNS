"""
-*- coding: utf-8 -*-
@Author : Smartpig
@Institution : DHU/DBLab
@Time : 2022/11/26 22:30
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import argparse
import os
from model import mymodel
from Load_data import gen_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score
from torch.optim.lr_scheduler import StepLR

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


seed = 6
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def parse_args():
    parser = argparse.ArgumentParser(description="请选择你的参数")
    #Parameters to split dataset
    parser.add_argument('--geneDataName', help='Gene expression data file name', type=str, default='ExpressionData.csv')
    parser.add_argument('--refNetworkName', help='Gene regulator network file name', type=str, default='refNetwork.csv')
    parser.add_argument('--datasetPath', help='Split folder file name', type=str, default='dataset')
    parser.add_argument('--k', help='k-fold cross validation', type=int, default=5)
    #Parameters to train model
    parser.add_argument('--batchSize', help="请选择你的batchSize", type=int, default=40960)
    parser.add_argument('--decodeDim', help="请选择你的编码器输出维度", type=int, default=256)
    parser.add_argument('--aggregator', help="请选择聚合函数", choices=['mean', 'lstm', 'max'], default='mean')
    parser.add_argument('--normalize', help="请选择编码器是否要标准化", type=bool, default=True)
    parser.add_argument('--topkratio', help="请选择topk池化参数", type=int, default=1)
    args = parser.parse_args()
    return args



def main(dataPath, args):
    os.chdir(dataPath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geneExpData = pd.read_csv(args.geneDataName, sep=',', header=0, index_col=0)
    x = torch.from_numpy(np.array(geneExpData, dtype=np.float64)).float()
    AUROC, AUPRC = [],[]
    for i in range(args.k):
        trueEdge, trainEdge, testEdge = gen_train_test_dataset(i, args.k, args.datasetPath + '/k-fold/')
        data = Data(x=x, edge_index=trueEdge).to(device)
        trainEdge = trainEdge.to(device)
        model = mymodel(data.x.size()[1], args.decodeDim, args.aggregator, args.normalize, args.topkratio,
                        data.x.size()[0]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
        for epoch in range(0, 100):
            loss = train(data, model, optimizer, scheduler, trainEdge)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        print("OK")
        score = predict(model, data, testEdge[:, 0:2], args.batchSize)
        trueEdge = np.array(testEdge[:, 2])
        auroc, auprc = comp_res(score, trueEdge)
        print(auroc, auprc)
        AUROC.append(auroc)
        AUPRC.append(auprc)
    return AUROC, AUPRC



def gen_train_test_dataset(i, k, path):
    train_df = pd.DataFrame(columns=['TF', 'Target', 'Label'])
    test_df = pd.read_csv(path + str(i) + '.csv', sep=',', header=0, index_col=0)
    for j in range(k):
        if j == i:
            continue
        train_df = train_df.append(pd.read_csv(path + str(j) + '.csv', sep=',', header=0, index_col=0))
    trueEdge = pd.DataFrame(train_df.loc[train_df['Label'] == 1, ['TF', 'Target']], dtype='int64')
    trueEdge = torch.from_numpy(np.array(trueEdge).T).long()
    trainEdge = torch.from_numpy(np.array(train_df.sample(frac=1), dtype='int64'))
    testEdge = torch.from_numpy(np.array(test_df.sample(frac=1), dtype='int64'))
    return trueEdge, trainEdge, testEdge



def train(data, model, optimizer, scheduler, trainEdge):
    model.train()
    # 梯度置零
    optimizer.zero_grad()
    x, topkarr = model.Sage_forword(data.x, data.edge_index)
    linkPredict = model.score(x, topkarr, trainEdge)
    linkLabels = trainEdge[:, 2].float()
    loss = F.binary_cross_entropy_with_logits(linkPredict, linkLabels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss



@torch.no_grad()
def predict(model, data, testEdge, batchSize):
    model.eval()
    inputx, topkarr = model.Sage_forword(data.x, data.edge_index)

    # relEmb = getRelEmb(data.edge_index, inputx).view(1, inputx.shape[1])

    Score = np.empty(shape=0)
    stack_list = []
    times = 1
    for i in range(testEdge.shape[0]):
        a = inputx[int(testEdge[i][0])].view(1, model.decodeDim)
        b = inputx[int(testEdge[i][1])].view(1, model.decodeDim)
        hrt = torch.cat([a, b, topkarr], 0)
        stack_list.append(hrt)
        if times % batchSize == 0:  # 表示需要切分
            x = torch.stack(stack_list, 0)
            batchScore = model.predict(x)
            Score = np.hstack((Score, batchScore))
            stack_list.clear()
        times = times + 1
    if stack_list:  # 不为空，表示还存在部分数据未输入
        x = torch.stack(stack_list, 0)
        batchScore = model.predict(x)
        # 分数追加
        Score = np.hstack((Score, batchScore))
        # 清空已有数据
        stack_list.clear()
    Score = Score.reshape(Score.shape[0], 1)
    return Score



def comp_res(score, trueEdge):
    reslist = []
    for i in range(trueEdge.size):
        Edge = [score[i], trueEdge[i]]
        reslist.append(Edge)
    resDF = pd.DataFrame(reslist, columns=['pre', 'true'])
    fpr, tpr, thresholds = roc_curve(resDF['true'], resDF['pre'], pos_label=1)
    auroc = auc(fpr, tpr)
    precision, recall, thresholds_PR = precision_recall_curve(resDF['true'], resDF['pre'])
    auprc = auc(recall, precision)
    # return auroc, auprc
    return auroc,auprc



if __name__ == '__main__':
    args = parse_args()
    path = 'E:/Project/My_GRN_Prediction/data'

    datafile = 'example'

    dataPath = path + '/' + datafile + '/'

    resDF = pd.DataFrame(columns=['AUROC', 'AUPRC', 'Density'])
    resDF.to_csv(dataPath + 'result.csv', index=True, sep=',')
    for files in os.listdir(dataPath):
        os.chdir(dataPath)
        if files == 'result.csv':
            continue
        for file in os.listdir(files):
            print(file+'_'+files)
            tempDataPath = dataPath + files + '/' + file + '/'
            density = gen_split(tempDataPath, args)
            AUROC, AUPRC = main(tempDataPath, args)
            resDF = pd.read_csv(dataPath+'result.csv', sep=',', index_col=0)
            resDF.loc[file+'_'+files] = [AUROC, AUPRC, density]
            # resDF = resDF.append(pd.DataFrame([file+'_'+files, AUROC, AUPRC, density]))
            resDF.to_csv(dataPath+'result.csv', index=True, sep=',')

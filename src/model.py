"""
-*- coding: utf-8 -*-
@Author : Smartpig
@Institution : DHU/DBLab
@Time : 2022/11/28 19:43
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TopKPooling


class mymodel(nn.Module):
    def __init__(self, encodeDim, decodeDim, aggregator, normalize, topkratio, nodeNum):
        super(mymodel, self).__init__()
        self.decodeDim = decodeDim
        self.cnnChannel = topkratio + 2 if topkratio >= 1 else (int)(topkratio * nodeNum + 2)

        self.encode = SAGEConv(encodeDim, decodeDim, aggregator, normalize)
        self.topkpool = TopKPooling(decodeDim, topkratio)

        self.bn1 = nn.BatchNorm1d(self.cnnChannel)
        self.bn2 = nn.BatchNorm1d(self.cnnChannel)
        self.bn3 = nn.BatchNorm1d(self.cnnChannel)
        self.cnn1 = nn.Conv1d(self.cnnChannel, self.cnnChannel, kernel_size=2, stride=1)
        self.cnn2 = nn.Conv1d(self.cnnChannel, self.cnnChannel, kernel_size=2, stride=1)
        self.cnn3 = nn.Conv1d(self.cnnChannel, self.cnnChannel, kernel_size=2, stride=1)
        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.pool3 = nn.MaxPool1d(2)

        # self.pool1 = nn.MaxPool1d(2)
        # self.pool2 = nn.MaxPool1d(2)
        # self.pool3 = nn.AvgPool1d(2)
        # self.pool4 = nn.AvgPool1d(2)


        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.sigmoid = nn.Sigmoid()


    def TopkPooling(self, x, edge_index):
        x, _, edge_index, _, _, _ = self.topkpool(x, edge_index)
        return x


    def Sage_forword(self, x, edge_index):
        x = self.encode(x, edge_index).to(x.device)
        x = F.relu(x)
        topkarr = self.TopkPooling(x, edge_index)
        return x, topkarr


    def creatDecodeInput(self, x, topkarr, trainEdge):
        stackList = []
        c = topkarr
        for i in range(trainEdge.shape[0]):
            a = x[int(trainEdge[i][0])].view(1, self.decodeDim)
            b = x[int(trainEdge[i][1])].view(1, self.decodeDim)
            hrt = torch.cat([a, b, c], 0)
            stackList.append(hrt)
        x = torch.stack(stackList, 0)
        return x


    def calmodel(self, x):
        # （1）
        # x = self.bn1(x)
        # x = F.relu(self.cnn1(x))
        # x = self.pool1(x)
        #
        # x = x.view(-1, x.size()[1] * x.size()[2])
        # x = F.dropout(x, training=self.training, p=0.5)
        # x = self.fc1(x)
        # x = self.bn4(x)
        # x = x + torch.randn_like(x) * 0.1
        # x = self.fc2(x)
        # score = self.sigmoid(x).view(-1)

        # # （2）
        # x1 = x
        # x1 = self.bn1(x1)
        # x1 = F.relu(self.cnn1(x1))
        # x1 = self.pool1(x1)
        #
        # x2 = self.bn2(x1)
        # x2 = F.relu(self.cnn2(x2))
        # x2 = self.pool2(x2)
        #
        # x3 = x
        # x3 = self.bn1(x3)
        # x3 = F.relu(self.cnn1(x3))
        # x3 = self.pool1(x3)
        #
        # x4 = self.bn1(x3)
        # x4 = F.relu(self.cnn1(x4))
        # x4 = self.pool1(x4)
        #
        # x = torch.cat([x1, x2, x3, x4], 2)
        # x = x.view(-1, x.size()[1] * x.size()[2])
        # x = F.dropout(x, training=self.training, p=0.5)
        # x = self.fc1(x)
        # x = self.bn4(x)
        # x = self.fc2(x)
        # score = self.sigmoid(x).view(-1)
        #

        x = self.bn1(x)
        x = F.relu(self.cnn1(x))
        x = self.pool1(x)
        x1 = x

        x = self.bn2(x)
        x = F.relu(self.cnn2(x))
        x = self.pool2(x)
        x2 = x

        x = self.bn3(x)
        x = F.relu(self.cnn3(x))
        x = self.pool3(x)
        x3 = x

        x = torch.cat([x1, x2, x3], 2)
        x = x.view(-1, x.size()[1] * x.size()[2])
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.fc2(x)
        score = self.sigmoid(x).view(-1)
        return score


    def score(self, x, topkarr, trainEdge):
        x = self.creatDecodeInput(x, topkarr, trainEdge)
        score = self.calmodel(x)
        return score


    def predict(self, x):
        """
        预测最终结果
        x表示解码器输入即为：hrt堆叠
        """
        # 预测最终分数
        score = self.calmodel(x)
        return score.cpu().data.numpy()

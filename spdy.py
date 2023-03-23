# SPDY search & DP algorithm implementation.


import copy
import math
import random

import numpy as np
import torch

from modelutils import *

# 定义一个Class 
class SPDY:
    # 初始化spdy类
    def __init__(
        self,
        target, db, errors, baselinetime, prunabletime, timings,
        get_model, run, dataloader,
        skip_layers=[], dpbuckets=10000
    ):
        self.target = target
        self.db = db
        self.run = run
        self.dpbuckets = dpbuckets

        self.modelp = get_model()
        self.layersp = find_layers(self.modelp)

        # 返回网上下载的模型的预测结果
        self.batches = []
        for batch in dataloader:
            self.batches.append(run(self.modelp, batch, retmoved=True))

        self.layers = list(db.layers())
        self.layers = [l for l in self.layers if l not in skip_layers]
        # 每层都有各种稀疏度等级
        self.sparsities = [list(errors[self.layers[l]].keys()) for l in range(len(self.layers))]
        # 稀疏度在所有稀疏等级中排序的平方
        self.costs = [
            [errors[self.layers[l]][s] for s in self.sparsities[l]] for l in range(len(self.layers))
        ]
        # 每层的时间
        self.timings = [
            [timings[self.layers[l]][s] for s in self.sparsities[l]] for l in range(len(self.layers))
        ]

        self.baselinetime = baselinetime
        self.prunabletime = prunabletime
        # 目标时间是基础时间的加速比分之一倍，减去通过剪枝可节省的时间（不可节省的时间指的是第一层最后一层，以及其他没有参数的时间）
        targettime = self.baselinetime / self.target - (self.baselinetime - self.prunabletime)
        # 每层中最小的时间总和
        best = sum(min(c) for c in self.timings)
        # 通过最好的时间计算最佳加速比
        if self.prunabletime < self.baselinetime:
            print('Max target:', self.baselinetime / (best + self.baselinetime - self.prunabletime))
        # 每个桶的大小 设置为目标时间除以桶的数量
        self.bucketsize = targettime / self.dpbuckets

        # 遍历所有层的时间（这是用来作什么的？）
        for row in self.timings:
            # 遍历所有稀疏度的时间
            for i in range(len(row)):
                row[i] = int(round(row[i] / self.bucketsize))

        print('Loss/Base:', self.get_loss(self.modelp))

    def dp(self, costs):
        DP = np.full((len(costs), self.dpbuckets + 1), float('inf'))
        PD = np.full((len(costs), self.dpbuckets + 1), -1)

        for sparsity in range(len(costs[0])):
            if costs[0][sparsity] < DP[0][self.timings[0][sparsity]]:
                DP[0][self.timings[0][sparsity]] = costs[0][sparsity]
                PD[0][self.timings[0][sparsity]] = sparsity
        for layer in range(1, len(DP)):
            for sparsity in range(len(costs[layer])):
                timing = self.timings[layer][sparsity]
                score = costs[layer][sparsity]
                if timing == 0:
                    tmp = DP[layer - 1] + score
                    better = tmp < DP[layer]
                    if np.sum(better):
                        DP[layer][better] = tmp[better]
                        PD[layer][better] = sparsity
                    continue
                if timing > self.dpbuckets:
                    continue
                tmp = DP[layer - 1][:-timing] + score
                better = tmp < DP[layer][timing:]
                if np.sum(better):
                    DP[layer][timing:][better] = tmp[better]
                    PD[layer][timing:][better] = sparsity

        score = np.min(DP[-1, :])
        timing = np.argmin(DP[-1, :])
        
        solution = []
        for layer in range(len(DP) - 1, -1, -1):
            solution.append(PD[layer][timing])
            timing -= self.timings[layer][solution[-1]]
        solution.reverse()
        return solution

    # 计算带有灵敏度的cost
    def gen_costs(self, coefs):
        return [
            [self.costs[i][j] * coefs[i] for j in range(len(self.costs[i]))] \
            for i in range(len(self.costs))
        ]

    def stitch_model(self, solution):
        model = copy.deepcopy(self.modelp)
        layers = find_layers(model)
        # 每层layer，对应第几个稀疏度等级
        config = {
            self.layers[i]: self.sparsities[i][solution[i]] for i in range(len(self.layers))
        }
        self.db.stitch(layers, config)
        return model

    # 运行模型，返回loss的平均值
    @torch.no_grad()
    def get_loss(self, model):
        loss = 0
        for batch in self.batches:
            loss += self.run(model, batch, loss=True)
        return loss / len(self.batches) 

    # 通过灵敏度向量，执行DP，得到solution，通过solution拼接模型，用于计算loss
    def get_score(self, coefs):
        costs = self.gen_costs(coefs)
        # solution 表示每个层的稀疏度等级
        solution = self.dp(costs)
        model = self.stitch_model(solution)
        return self.get_loss(model)

    def save_profile(self, coefs, filename=''):
        solution = self.dp(self.gen_costs(coefs))
        if filename:
            with open(filename, 'w') as f:
                for i in range(len(solution)):
                    f.write('%s %s\n' % (self.sparsities[i][solution[i]], self.layers[i]))
        else:
            for i in range(len(solution)):
                print('%s %s' % (self.sparsities[i][solution[i]], self.layers[i]))

    def score(self, filename):
        tmp = []
        with open(filename, 'r') as f:
            solution = []
            for i, l in enumerate(f.readlines()):
                splits = l.split(' ')
                sparsity = splits[0]
                tmp.append(float(sparsity))
                name = splits[1][:-1]
                j = self.sparsities[i].index(sparsity)
                solution.append(j)

        print('Speedup:', self.baselinetime / (
            self.baselinetime - self.prunabletime + \
            sum(t[s] for s, t in zip(solution, self.timings)) * self.bucketsize
        ))

        model = self.stitch_model(solution)
        print('Loss/Pruned:', self.get_loss(model))
        return model

    def dpsolve(self, save=''):
        coefs = np.ones(len(self.layers))
        print('Loss/Pruned:', self.get_score(coefs))
        self.save_profile(coefs)
        if save:
            self.save_profile(coefs, save)

    # 寻找合适的profile
    def search(
        self, save='', randinits=100, maxnoimp=100, layerperc=.1
    ):
        evals = 0
        print('Finding init ...')
        # 灵敏度向量c
        coefs = None
        score = float('inf')
        for _ in range(randinits):
            # 随机初始化灵敏度向量
            coefs1 = np.random.uniform(0, 1, size=len(self.layers))
            score1 = self.get_score(coefs1)
            evals += 1
            print('%04d  %.4f %.4f' % (evals, score, score1))
            # 如果分数越小，则替换灵敏度向量
            if score1 < score:
                score = score1
                coefs = coefs1
        print('Running local search ...')
        for resamplings in range(round(layerperc * len(self.layers)), 0, -1):
            print('Trying %d resamplings ...' % resamplings)
            improved = True
            while improved: 
                improved = False
                for _ in range(maxnoimp):
                    coefs1 = coefs.copy()
                    for _ in range(resamplings):
                        coefs1[random.randint(0, len(self.layers) - 1)] = np.random.uniform(0, 1)
                    score1 = self.get_score(coefs1)
                    evals += 1
                    print('%04d  %.4f %.4f' % (evals, score, score1))
                    if score1 < score:
                        score = score1
                        coefs = coefs1
                        improved = True
                        break
        self.save_profile(coefs)
        if save:
            self.save_profile(coefs, save)


if __name__ == '__main__':
    import argparse

    from database import *
    from datautils import *

    parser = argparse.ArgumentParser()

    # 模型的名字
    parser.add_argument(
        'model', type=str, choices=get_models,
        help='Model to work with.'
    )
    # 数据集的名字
    parser.add_argument(
        'dataset', type=str, choices=DEFAULT_PATHS,
        help='Dataset to use.'
    )
    # 用于存储每层且每种稀疏度对应的重构权重的数据库。
    parser.add_argument(
        'database', type=str,
        help='Database location.'
    )
    # 记录每层时间的文件位置
    parser.add_argument(
        'timings', type=str,
        help='Timings file.'
    )
    # 目标加速比的大小
    parser.add_argument(
        'target', type=float,
        help='Target speedup.'
    )
    # 结果 profile 的位置。
    parser.add_argument(
        'profile', type=str,
        help='Where to save the resulting profile.'
    )

    # 数据集的位置
    parser.add_argument(
        '--datapath', type=str, default='',
        help='Path to dataset.'
    )
    # 随机的种子
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Seed to use for calibration set selection.'
    )
    # 校准数据集的样本数量
    parser.add_argument(
        '--nsamples', type=int, default=1024,
        help='Number of samples in the calibration dataset.'
    )

    # 解析输入的参数
    args = parser.parse_args()
    # 选择model（最原始的）和数据集
    get_model, test, run = get_functions(args.model)
    dataloader, testloader = get_loaders(args.dataset, noaug=True, nsamples=args.nsamples)

    # 初始化模型和数据库
    model = get_model()
    db = UnstrDatabase(args.database, model, skip=firstlast_names(args.model))

    # errors只有稀疏度相关，与具体权值无关
    errors = db.get_errors()
    baselinetime, prunabletime, timings = db.get_timings(args.timings)
    # 根据参数搜索合适的profile
    spdy = SPDY(
        args.target, db, errors, baselinetime, prunabletime, timings,
        get_model, run, dataloader
    )

    PROFILE_FILE = args.profile
    spdy.search(PROFILE_FILE)

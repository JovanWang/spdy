# Helper code for handling a the reconstruction database.


from collections import * 

import torch
import torch.nn as nn

from modelutils import *


# 数据库
class UnstrDatabase:
    # 根据文件名字初始化数据库。
    # 每层每个稀疏数值都有一个值。
    def __init__(self, path, model, skip=[]):
        self.db = defaultdict(OrderedDict)
        denselayers = find_layers(model)
        dev = next(iter(denselayers.values())).weight.device
        # 遍历所有模型文件，给每层和稀疏度赋值对应的权重张量
        for f in os.listdir(path):
            sparsity = '0.' + f.split('.')[0]
            sd = torch.load(os.path.join(path, f), map_location=dev)
            for layer in denselayers:
                if layer not in skip:
                    self.db[layer][sparsity] = sd[layer + '.weight']

    def layers(self):
        return list(self.db.keys())

    # 加载稀疏度对应的权重矩阵到layer上
    def load(self, layers, name, config='', sd=None):
        if sd is not None:
            layers[name].weight.data = sd[name + '.weight']
            return
        layers[name].weight.data = self.db[name][config]

    # 将config中每层的稀疏度与模型的layer对应上
    def stitch(self, layers, config):
        for name in config:
            self.load(layers, name, config[name])

    # 解析profile中name对应的稀疏度，进一步将稀疏度对应到权重矩阵，并将其缝补到layer上，得到新的模型
    def load_file(self, model, profile):
        config = {}
        with open(profile, 'r') as f:
            for line in f.readlines():
                splits = line.split(' ')
                sparsity = splits[0]
                name = splits[1][:-1]
                config[name] = sparsity
        for name in self.db:
            if name not in config:
                config[name] = '0.0000'
        layers = find_layers(model)
        self.stitch(layers, config)

    # 计算errors[layer][sparse]，只与稀疏度相关
    def get_errors(self):
        errors = {}
        for name in self.db:
            errors[name] = {}
            # 稀疏度从小到大排序，稀疏度越小，排名越前；则i更小，error更小
            for i, s in enumerate(sorted(self.db[name])):
                # i 是稀疏度排在所有稀疏度里大小的第几位。s是稀疏度值本身
                errors[name][s] = (i / (len(self.db[name])- 1)) ** 2
        return errors 

    # 
    def get_params(self, layers):
        tot = 0
        res = {}
        for name in layers:
            res[name] = {}
            tot += layers[name].weight.numel()
            for sparsity in self.db[name]:
                res[name][sparsity] = torch.sum(
                    (self.db[name][sparsity] != 0).float()
                ).item()
        return tot, tot, res

    # 返回整体计算时间，以及每层每种稀疏度级别的时间
    def get_timings(self, path):
        timings = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            # 第二行和第四行分别表示基础时间和剪枝后的时间
            baselinetime = float(lines[1])
            prunabletime = float(lines[3])
            i = 4
            # 花式遍历所有行
            while i < len(lines):
                # 第五行是layer的name
                name = lines[i].strip()
                timings[name] = {}
                i += 1
                # 一直遍历包含‘ ’的行。得到每个稀疏级别对应的执行的时间
                while i < len(lines) and ' ' in lines[i]:
                    time, level = lines[i].strip().split(' ')
                    timings[name][level] = float(time)
                    i += 1
        timings = {l: timings[l] for l in self.layers()}
        return baselinetime, prunabletime, timings

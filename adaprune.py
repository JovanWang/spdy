# AdaPrune & global AdaPrune implementations for unstructured, blocked and N:M pruning.


import copy
import math
import torch
import torch.nn as nn

# 层基类
class MagLayerPruner: 

    # Assumes that all 0s have already been pruned
    def __init__(self, layer, sparsity, lr=1e-3):
        self.layer = layer
        # 按该层参数的总数的百分比稀疏构造掩码
        # tmp是排序后tensor的值
        tmp = torch.sort(torch.abs(self.layer.weight.data.reshape(-1)))[0]
        # 该层所有权重的乘积乘以稀疏度取整得到一个阈值坐标，通过这个坐标从tmp中取出对应的值作为阈值。
        thresh = tmp[int(self.layer.weight.numel() * sparsity)]
        self.mask = torch.abs(self.layer.weight.data) > thresh
        self.apply_mask()
        self.optim = torch.optim.Adam([self.layer.weight], lr=lr)

    # 计算输入在该层上的临时输出与输出之间的差值平方占原输出的值平方进行比值。作为loss更新梯度，更新后再次应用该层的mask。
    def optim_step(self, inp, out):
        # 输出的二范数的平方，也就是所有输出元素的平方和
        norm = torch.norm(out).item() ** 2
        # 通过该层计算得到输出，并与out相减再平方。
        out1 = self.layer(inp)
        out1.sub_(out)
        out1.pow_(2)
        # 统计out1的差值的平方和 和 norm 的除数关系 作为loss。
        loss = torch.sum(out1) / norm
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()  
        self.apply_mask()

    def apply_mask(self):
        self.layer.weight.data *= self.mask

# 层剪枝类
class NM50LayerPruner(MagLayerPruner):

    # Assume number of weights in layer is divisible by blocksize
    def __init__(self, layer, blocksize, lr=1e-3):
        # 有权重的除了第一层和最后一层外的所有layer中的一个
        self.layer = layer
        w = self.layer.weight.data
        # 如果是卷积层，则从 size (N, C, H, W) 切换为 （N, H, W, C）
        if len(w.shape) == 4:
            w = w.permute(0, 2, 3, 1)
        # 权重矩阵w按 blocksize 为列主的大小展开。然后取绝对值最大的前一半值，得到其索引对应的位置i。
        _, i = torch.topk(
            torch.abs(w.reshape((-1, blocksize))), blocksize // 2, dim=1
        )
        # 按照w按 blocksize 为列主的大小展开构造 全0掩码矩阵
        self.mask = torch.zeros_like(w).reshape(-1, blocksize)
        # 根据索引 i 将掩码矩阵对应位置改为1
        for j in range(blocksize // 2):
            self.mask[torch.arange(self.mask.shape[0]), i[:, j]] = 1 
        # 将掩码矩阵改变成w权重矩阵的形状
        self.mask = self.mask.reshape(w.shape)
        # 将mask改回成 (N, C, H, W) 的排列模式
        if len(w.shape) == 4:
            self.mask = self.mask.permute(0, 3, 1, 2)
        # mask从0/1变成一个true/false 的矩阵
        self.mask = self.mask == 1
        # 权重w与mask点乘
        self.apply_mask()
        # 优化器是Adam，包含了权重和学习率
        self.optim = torch.optim.Adam([self.layer.weight], lr=lr)

class BlockLayerPruner(MagLayerPruner):

    # Assume number of weights in layer is divisible by blocksize
    def __init__(self, layer, blocksize, sparsity, lr=1e-3):
        self.layer = layer
        w = self.layer.weight.data
        if len(w.shape) == 4:
            w = w.permute(0, 2, 3, 1)
        # 记录每个block的元素和。
        tmp = torch.sum(torch.abs(w.reshape((-1, blocksize))), 1)
        # 所有block和的乘积乘以稀疏度取整得到一个阈值坐标，通过这个坐标从tmp中取出对应的值作为阈值。
        thresh = torch.sort(tmp)[0][int(tmp.numel() * sparsity)]
        self.mask = torch.zeros_like(w).reshape(-1, blocksize)
        self.mask[tmp > thresh, :] = 1
        self.mask = self.mask.reshape(w.shape)
        if len(w.shape) == 4:
            self.mask = self.mask.permute(0, 3, 1, 2)
        self.mask = self.mask == 1
        self.apply_mask()
        self.optim = torch.optim.Adam([self.layer.weight], lr=lr)


# Assume that we only prune the weight `parameter` of each layer
# Assume that `modelp` and `modeld` are on the same GPU
# Assume models are in eval mode

# 传入一个全新的模型用来做推理，使用hook绑定新模型的输入输出和我们要更新的原模型的参数。
def layerw_adaprune(
    pruners, modeld, dataloader, run, iters=10
):
    # 全新的返回所有的卷积层和线性层存入layersp中（用来跑钩子）
    layersd = find_layers(modeld)

    # hook的使用场景在于，我们直接使用pytoch封装好的模型时，获取其中的输出信息。
    def hook(name):
        def tmp(layer, inp, out):
            with torch.enable_grad():
                pruners[name].optim_step(inp[0].data, out.data)
        return tmp

    handles = []
    # Registers a global forward hook for all the modules
    # The hook will be called every time after forward() has computed an output. 
    # 为每个层增加前向计算的hook，每次forward都要被调用。
    # handle里面装的是全新层的hook的操作记录，即用每层完整的输出与mask后的输出的差值去更新以前层的参数。
    for name in pruners:
        handles.append(layersd[name].register_forward_hook(hook(name)))
    # 将每层的参数都移到Device上
    dev = layersd[next(iter(layersd))].weight.device
    # 重复10次，使用全新的modeld运行整个数据集，便于更新pruners中的参数。
    for i in range(iters):
        # print(i)
        for batch in dataloader:
            with torch.no_grad():
                run(modeld, batch)
    for h in handles:
        h.remove()

# 原始模型和逐层更新后的模型都传入了。
def global_adaprune(
    pruners, modelp, modeld, dataloader, run,
    iters=100, lr=1e-5
):
    # 分别获取逐层更新后的模型 和 原始模型的layers
    layersp = find_layers(modelp) 
    layersd = find_layers(modeld)

    # 钩子的作用就是返回该层的输出结果。
    def cache_output(name, outputs):
        def tmp(layer, inp, out):
            outputs[name] = out
        return tmp
    outputsp = {}
    handlesp = []
    for name in layersp:
        handlesp.append(
            layersp[name].register_forward_hook(cache_output(name, outputsp))
        )
    outputsd = {}
    handlesd = []
    for name in layersd:
        handlesd.append(
            layersd[name].register_forward_hook(cache_output(name, outputsd))
        )

    dev = layersp[next(iter(layersp))].weight.device
    criterion = nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(modelp.parameters(), lr=lr)

    # 
    for i in range(iters):
        cumloss = 0
        for batch in dataloader:
            # 原始的不带梯度，更新后的带梯度进行run。
            with torch.no_grad():
                run(modeld, batch)
            run(modelp, batch)
            # 根据结果计算loss，原始的每层的计算结果和更新后的结果的差距的平方值与原始值得比值
            loss = 0
            for name in outputsd:
                norm = torch.norm(outputsd[name].data).item() ** 2
                loss += criterion(outputsp[name], outputsd[name].data) / norm
            cumloss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
            for p in pruners.values():
                p.apply_mask()
        print('%05d: %.6f' % (i, cumloss / len(dataloader)))

    for h in handlesp:
        h.remove()
    for h in handlesd:
        h.remove()


if __name__ == '__main__':
    import argparse
    import os

    from datautils import *
    from modelutils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str, choices=get_models,
        help='Model to work with.'
    )
    parser.add_argument(
        'dataset', type=str, choices=DEFAULT_PATHS,
        help='Dataset to use.'
    )
    # 要执行的操作：（NM剪枝，生成数据库，加载评估profile）
    parser.add_argument(
        'mode', type=str, choices=['nmprune', 'gen', 'load'],
        help='Operation mode of the script; "nmprune" for N:M pruning, "gen" for database generation, and "load" for profile evaluation.'
    )

    # 存放数据库的文件夹位置
    parser.add_argument(
        '--collect_to', type=str, default='',
        help='Folder to store database in; only used in "gen" mode.'
    )
    # 加载层重构数据集的文件夹位置
    parser.add_argument(
        '--stitch_from', type=str, default='',
        help='Folder to load database from; only used in "load" mode.'
    )
    # profile文件的名字
    parser.add_argument(
        '--profile', default='',
        help='Profile to load; only used in "load" mode.'
    )
    # 是否以及在何处保存生成的检查点，不能在gen模式下使用
    parser.add_argument(
        '--save', default='',
        help='Whether and where to save the resulting checkpoint; not used in "gen" mode.'
    )

    # NM剪枝的块大小
    parser.add_argument(
        '--nmblocksize', type=int, default=4,
        help='Blocksize for N:M pruning.'
    )
    # 块剪枝的块大小
    parser.add_argument(
        '--blocksize', type=int, default=4,
        help='Blocksize used for block pruning.'
    )

    # 数据集的位置
    parser.add_argument(
        '--datapath', type=str, default='',
        help='Path to dataset.'
    )
    # 随机种子
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Seed to use for calibration set selection.'
    )
    # 验证集采样的数量
    parser.add_argument(
        '--nsamples', type=int, default=1024,
        help='Number of samples in the calibration dataset.'
    )

    # Database最小稀疏度。
    parser.add_argument(
        '--min-sparsity', type=float, default=.4,
        help='Minimum database sparsity.'
    )
    # Database最大稀疏度。
    parser.add_argument(
        '--max-sparsity', type=float, default=.99,
        help='Maximum database sparsity.'
    )
    # 最小和最大稀疏度之间的相等相对步数
    parser.add_argument(
        '--steps', type=int, default=40,
        help='Number of equal relative steps between min and max sparsity.'
    )

    # 分层 AdaPrune 和 全局AdaPrune剪枝的批大小
    parser.add_argument(
        '--batchsize', type=int, default=32,
        help='AdaPrune and global AdaPrune batchsize.'
    )
    # 分层 AdaPrune 的数据集传递次数
    parser.add_argument(
        '--iters_layerw', type=int, default=10,
        help='Number of dataset passes for layer-wise AdaPrune.'
    )
    # 全局 AdaPrune 的数据集传递次数
    parser.add_argument(
        '--iters_global', type=int, default=100,
        help='Number of dataset passes for global AdaPrune.'
    )
    # 分层Adaprune的学习率
    parser.add_argument(
        '--lr_layerw', type=float, default=1e-3,
        help='Learning rate for layer-wise AdaPrune.'
    )
    # 全局Adaprunne的学习率
    parser.add_argument(
        '--lr_global', type=float, default=1e-5,
        help='Learning rate for global AdaPrune.'
    )

    args = parser.parse_args()

    # 加载数据集
    dataloader, testloader = get_loaders(
        args.dataset, path=args.datapath,
        nsamples=args.nsamples, seed=args.seed,
        batchsize=args.batchsize
    )
    get_model, test, run = get_functions(args.model)

    # 返回下载后的模型
    modelp = get_model()
    modeld = get_model()

    # 返回所有的卷积层和线性层存入layersp中
    layersp = find_layers(modelp)
    print("layersp: ", layersp)
    pruners = {}

    if args.mode == 'load':
        from database import *
        # 初始化db数据库
        db = UnstrDatabase(args.stitch_from, modelp)
        # 将profile中对应的配置（每层的稀疏度），及其对应的权重张量，加载到模型p中
        db.load_file(modelp, args.profile)

        pruners = {}
        # 根据配置得到的权重文件计算得到不同层的稀疏度，并用于初始化 MagLayerPruner。
        for name in layersp:
            pruners[name] = MagLayerPruner(
                layersp[name],
                # 判断权重矩阵中，多少个0，然后除以总数，得到稀疏度。
                torch.mean((layersp[name].weight == 0).float()).item(),
                lr=args.lr_layerw
            )
        # 测试结果
        test(modelp, testloader)
        if args.iters_global > 0:
            global_adaprune(
                pruners, modelp, modeld, dataloader, run, iters=args.iters_global, lr=args.lr_global
            )
            test(modelp, testloader)

        # 保存profile对应的模型
        if args.save:
            torch.save(modelp.state_dict(), args.save)
        exit()

    if args.mode == 'gen':
        # 检查生成数据库路径是否存在
        if not os.path.exists(args.collect_to):
            os.makedirs(args.collect_to)

        params = []
        for n, p in modelp.named_parameters():
            # 只记录有有效权重的参数名字，并将其weight去掉
            if ('weight' not in n) or (len(p.shape) == 1):
                continue
            params.append(n.replace('.weight', ''))

        # 把模型移到CPU上，保存一下初始参数。
        modelp = modelp.cpu()
        torch.save(modelp.state_dict(), os.path.join(args.collect_to, '0000.pth'))
        modelp = modelp.to(DEV)

        # 稠密度初始值为 1-最小稠密度。
        density = 1 - args.min_sparsity
        # 例如：0.6连乘10次某个值得到0.8，delta就是这个值
        delta = ((1 - args.max_sparsity) / density) ** (1 / args.steps)
        # 遍历每一个稀疏度的Step
        for _ in range(args.steps + 1):
            print('%.4f' % (1 - density))
            # 遍历每个层，块大小大于1用块剪枝，小于1，则用原始剪枝。裁剪比例用 1 - density 控制
            for name in params:
                if args.blocksize > 1:
                    pruners[name] = BlockLayerPruner(layersp[name], args.blocksize, 1 - density, lr=args.lr_layerw)
                else:
                    pruners[name] = MagLayerPruner(layersp[name], 1 - density, lr=args.lr_layerw)
            # 分层剪枝，并保存模型参数
            layerw_adaprune(pruners, modeld, dataloader, run, iters=args.iters_layerw)
            modelp = modelp.cpu()
            torch.save(
                modelp.state_dict(),
                os.path.join(
                    args.collect_to, '%s.pth' % ('%.4f' % (1 - density))[2:]
                )
            )
            modelp = modelp.to(DEV)
            density *= delta
        exit()

    # nmprune 模型
    if args.mode == 'nmprune':
        params = []
        for n, p in modelp.named_parameters():
            # print("n:", n)
            # n, p : conv1.weight Parameter containing: tensor([[[[]]]])
            # n, p : bn1.bias Parameter containing: tensor([])
            # 只记录有有效权重的参数名字，并将其weight去掉
            if ('weight' not in n) or (len(p.shape) == 1):
                continue
            params.append(n.replace('.weight', ''))
        # 去掉第一层和最后一层
        params = [p for p in params if p not in firstlast_names(args.model)]

        # 把每个层都包装成一个NM50LayerPruner类，并传入块大小 和 学习率
        for name in params:
            # 初始化操作：得到每层按50%稀疏掩码矩阵
            pruners[name] = NM50LayerPruner(layersp[name], args.nmblocksize, lr=args.lr_layerw)

        # 传入有掩码的每层的参数，全新的模型，数据集，一个batch的计算函数，迭代次数。
        # 用于更新每层的参数
        layerw_adaprune(pruners, modeld, dataloader, run, iters=args.iters_layerw)
        # 测试每层参数都被更新后的结果。
        test(modelp, testloader)
        # 判断是否启动了全局更新
        if args.iters_global:
            global_adaprune(
                pruners, modelp, modeld, dataloader, run, iters=args.iters_global, lr=args.lr_global
            )
            # 测试全局更新后的记过
            test(modelp, testloader)
        # 保存模型结果
        if args.save:
            torch.save(modelp.state_dict(), args.save)

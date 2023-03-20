import os
import sys
sys.path.append('yolov5')

import torch
import torch.nn as nn


DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def test(model, dataloader):
    train = model.training
    model.eval()
    print('Evaluating ...')
    dev = next(iter(model.parameters())).device
    preds = []
    ys = []
    for x, y in dataloader:
        preds.append(torch.argmax(model(x.to(dev)), 1))
        ys.append(y.to(dev))
    acc = torch.mean((torch.cat(preds) == torch.cat(ys)).float()).item()
    acc *= 100
    print('acc: %.2f' % acc)
    if train:
        model.train()

@torch.no_grad()
def test_yolo(model, dataloader):
    import json
    from pathlib import Path
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from yolov5.utils.general import coco80_to_coco91_class, non_max_suppression, scale_coords, xywh2xyxy
    from yolov5.utils.metrics import ap_per_class
    from yolov5.val import process_batch, save_one_json

    train = model.training
    model.eval()
    print('Evaluating ...')
    dev = next(iter(model.parameters())).device

    conf_thres = .001
    iou_thres = .65

    iouv = torch.Tensor([.5]) 
    niou = iouv.numel()
    class_map = coco80_to_coco91_class()
    jdict = []
    names = {k: v for k, v in enumerate(model.names)}

    for i, (im, targets, paths, shapes) in enumerate(dataloader): 
        im = im.to(dev)
        targets = targets.to(dev)
        out, _ = model(im)
        nb, _, height, width = im.shape
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(dev)
        out = non_max_suppression(
            out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False
        )
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path, shape = Path(paths[si]), shapes[si][0]
            if len(pred) == 0:
                continue
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])
            save_one_json(predn, jdict, path, class_map)

    anno_json = dataloader.dataset.original.path.replace(
        'images/val2017', 'annotations/instances_val2017.json'
    )
    import random
    pred_json = 'yolo-preds-for-eval-%d.json' % random.randint(0, 10 ** 6)
    with open(pred_json, 'w') as f:
        json.dump(jdict, f)

    anno = COCO(anno_json)
    pred = anno.loadRes(pred_json)
    eval = COCOeval(anno, pred, 'bbox')
    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.original.img_files]
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    map, map50 = eval.stats[:2]
    print(100 * map50)
    os.remove(pred_json)

    if train:
        model.train()

# 根据model名字判断，使用那个test函数
def get_test(name):
    if 'yolo' in name:
        return test_yolo
    return test

# 执行一个batch的模型计算
def run(model, batch, loss=False, retmoved=False):
    dev = next(iter(model.parameters())).device
    if retmoved:
        return (batch[0].to(dev), batch[1].to(dev))
    out = model(batch[0].to(dev))
    if loss:
        return nn.functional.cross_entropy(out, batch[1].to(dev)).item()
    return out

def run_yolo(model, batch, loss=False, retmoved=False):
    dev = next(iter(model.parameters())).device
    if retmoved:
        return (batch[0].to(dev), batch[1].to(dev))
    out = model(batch[0].to(dev))
    if not model.training:
        out = out[1]
    if loss:
        return model.computeloss(out, batch[1].to(dev))[0].item()
    return torch.cat([o.flatten() for o in out])

# 根据模型名字是否有yolo判断使用哪个run
def get_run(model):
    if 'yolo' in model:
        return run_yolo
    return run

# yolo模型的调用
def get_yolo(var):
    from yolov5.models.yolo import Model
    from yolov5.utils.downloads import attempt_download
    weights = attempt_download(var + '.pt')
    ckpt = torch.load(weights, map_location=DEV)
    model = Model(ckpt['model'].yaml)
    csd = ckpt['model'].float().state_dict()
    model.load_state_dict(csd, strict=False)
    from yolov5.utils.loss import ComputeLoss
    model.hyp = {
        'box': .05, 'cls': .5, 'cls_pw': 1., 'obj': 1., 'obj_pw': 1., 'fl_gamma': 0., 'anchor_t': 4
    }
    model = model.to(DEV)
    model.computeloss = ComputeLoss(model)
    return model


from torchvision.models import *

# 模型来自于 torchvision.models 模块，Resnet可以直接调用，yolo需要额外处理一下。
get_models = {
    # 'rn18': lambda: resnet18(pretrained=True),
    'rn18': lambda: resnet18(weights=ResNet18_Weights.DEFAULT),
    'rn34': lambda: resnet34(weights=ResNet34_Weights.DEFAULT),
    'rn50': lambda: resnet50(weights=ResNet50_Weights.DEFAULT),
    'rn101': lambda: resnet101(weights=ResNet101_Weights.DEFAULT),
    '2rn50': lambda: wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT),
    'yolov5s': lambda: get_yolo('yolov5s'),
    'yolov5m': lambda: get_yolo('yolov5m')
}

# 从一个预定好的list中返回对以ing的模型类，并移到GPU上，并切换成评估模式。
def get_model(model):
    model = get_models[model]()
    model = model.to(DEV)
    model.eval()
    return model

# 由adaprune.py文件调用，根据model名字，返回一组值model类，test，run
def get_functions(model):
    return lambda: get_model(model), get_test(model), get_run(model)

# 返回模型的最开始的层的名字 和 最后一层的层名字
def firstlast_names(model):
    if 'rn' in model:
        return ['conv1', 'fc']
    if 'yolo' in model:
        return ['model.0.conv']

import argparse
import os
import pickle
import random
from argparse import Action, FileType
from filelock import SoftFileLock
from collections import OrderedDict
from copy import copy
from itertools import chain
from random import choices, shuffle
from typing import Text, Union, Type, Any, Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.qat as nnqat
import torch.utils.data
from torch import Tensor
from torch.quantization import get_qat_module_mappings
arg_names = []


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="CIFAR-10/AlexNet/Quantization/Training/WeightFaults with PyTorch")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
arg_names.append('lr')
parser.add_argument('--epoch', default=300, type=int, help='number of epochs tp train for')
arg_names.append('epoch')
parser.add_argument('--trainBatchSize', default=128, type=int, help='training batch size')
parser.add_argument('--testBatchSize', default=128, type=int, help='testing batch size')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=str2bool, help='whether cuda is in use')
parser.add_argument('--prefix', type=str, default='', help='prefix for output file paths')
arg_names.append('prefix')
parser.add_argument('--dropout_p', type=float, default=0.5, help='probability of dropout')
arg_names.append('dropout_p')
parser.add_argument('--last_dropout', type=str2bool, default=True, help='whether enable last dropout')
arg_names.append('last_dropout')
parser.add_argument('--first_dropout', type=str2bool, default=False, help='whether enable first dropout')
arg_names.append('first_dropout')
parser.add_argument('--sigma', type=float, default=0.1, help='sigma used for randomized smoothing')
arg_names.append('sigma')
parser.add_argument('--train', type=str2bool, default=True, help='whether to train')
parser.add_argument('--berr', type=float, default=0.01, help='bit error rate to analyze')
arg_names.append('berr')
parser.add_argument('--t_mapping', type=str, default='None',
                    help='name of quantization mapping for training, find later in the code')
parser.add_argument('--e_mapping', type=str, default='None',
                    help='name of quantization mapping for evaluation, find later in the code')
arg_names.append('t_mapping')
arg_names.append('e_mapping')
parser.add_argument('--weight_clip', type=str2bool, default=True,
                    help='whether to constrain weights during training to [-0.1, 0.1]')
arg_names.append('weight_clip')
parser.add_argument('--mc_dropout', type=str2bool, default=False,
                    help='whether to evaluate in monte carlo mode dropout')
arg_names.append('mc_dropout')
args = parser.parse_args()

os.makedirs('weights', exist_ok=True)
os.makedirs('results', exist_ok=True)


def serialize_params(args, exclude=(), exclude_defaults=None):
    if exclude_defaults is None:
        exclude_defaults = {}
    return '-'.join('{}:{}'.format(a, getattr(args, a)) for a in arg_names if (
            a not in exclude and
            not (a in exclude_defaults and exclude_defaults[a] == getattr(args, a))
    ))


def deserialize_params(params):
    args = lambda: None
    for key_value in params.split('-'):
        key, value = key_value.split(':')
        setattr(args, key, value)
    return args


class InjectionMixin:

    enabled = True

    def init_injection(self):
        self.k = 0
        self.fault_stride = 3
        self.prob = self.weight.flatten().shape[0]
        self.grad_sum = torch.zeros(self.weight.shape)

    def weight_fake_quant_inject(self, fake_quant, weight):
        if self.enabled:
            weight = self.manipulate_float(weight)
        repeat = weight.flatten().shape[0] // fake_quant.scale.flatten().shape[0]
        scale = torch.repeat_interleave(fake_quant.scale, repeat).reshape(weight.shape)
        zero_point = torch.repeat_interleave(fake_quant.zero_point, repeat).reshape(weight.shape)
        quantized = torch.clamp(
            torch.round(
                weight / scale + zero_point),
            fake_quant.quant_min,
            fake_quant.quant_max)
        if self.enabled:
            quantized = self.manipulate_quantized(quantized)
        return (quantized - zero_point) * scale

    def manipulate_float(self, weight):
        return weight

    def manipulate_quantized(self, quantized):
        return quantized


class RandomBET(InjectionMixin):
    berr = args.berr

    def manipulate_quantized(self, signed_quantized):
        additive = torch.zeros(signed_quantized.shape, device=signed_quantized.device)
        for bit_magnitude in (1, 2, 4, 8, 16, 32, 64, 127):
            quantized = signed_quantized + 128
            mask = torch.rand(quantized.shape, device=quantized.device) > self.berr
            flip_sign = torch.masked_fill(- (torch.floor(quantized / bit_magnitude) % 2 - 0.5) * 2, mask, 0)
            additive += flip_sign * bit_magnitude
        return (quantized + additive) - 128


class RowHammerSprayAttack(InjectionMixin):
    berr = args.berr

    def manipulate_quantized(self, signed_quantized):
        additive = torch.zeros(signed_quantized.shape, device=signed_quantized.device)
        for bit_magnitude in (1, 2, 4, 8, 16, 32, 64, 127):
            quantized = signed_quantized + 128
            mask = torch.rand(quantized.shape, device=quantized.device) > self.berr
            flip_sign = torch.masked_fill(- (torch.floor(quantized / bit_magnitude) % 2 - 0.5) * 2, mask, 0)
            additive += flip_sign * bit_magnitude
        return (quantized + additive) - 128


class RowHammerUpSprayAttack(InjectionMixin):
    berr = args.berr

    def manipulate_quantized(self, signed_quantized):
        additive = torch.zeros(signed_quantized.shape, device=signed_quantized.device)
        for bit_magnitude in (1, 2, 4, 8, 16, 32, 64, 127):
            quantized = signed_quantized + 128
            mask = torch.rand(quantized.shape, device=quantized.device) > self.berr
            flip_sign = torch.masked_fill(- (torch.floor(quantized / bit_magnitude) % 2 - 0.5) * 2, mask, 0)
            flip_sign = torch.max(flip_sign, torch.zeros(flip_sign.shape, device=quantized.device))
            additive += flip_sign * bit_magnitude
        return (quantized + additive) - 128


class BlindRowHammerAttack(InjectionMixin):

    def manipulate_quantized(self, signed_quantized):
        raise DeprecationWarning
        if self.k == 0:
            return signed_quantized
        quantized = signed_quantized + 128
        indices = []
        offsets = list(range(self.fault_stride))
        shuffle(offsets)
        size = 1
        for d in quantized.shape:
            size *= d
        indices = choices(range(size), k=self.k)
        # for o in offsets:
        #     if len(indices) == self.k:
        #         break
        #     elif len(indices) > self.k:
        #         assert False
        #     indices.append(o)
        #     while len(indices) < self.k:
        #         to_append = indices[-1] + self.fault_stride
        #         if to_append >= size:
        #             break
        #         else:
        #             indices.append(to_append)
        mask = torch.scatter(torch.ones(size), 0,
                             torch.LongTensor(indices), 0).view(quantized.shape)
        bit_magnitude = 128
        flip_sign = torch.masked_fill(- (torch.floor(quantized / bit_magnitude) % 2 - 0.5) * 2, mask == 1, 0)
        additive = flip_sign * bit_magnitude
        return (quantized + additive) - 128


class InjectionConv2d(nnqat.Conv2d, InjectionMixin):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         qconfig)
        self.init_injection()

    def forward(self, input):
        return self._conv_forward(
            input,
            self.weight_fake_quant_inject(
                self.weight_fake_quant,
                self.weight_fake_quant(self.weight)))


class InjectionLinear(nnqat.Linear, InjectionMixin):

    def __init__(self, in_features, out_features, bias=True, qconfig=None):
        super().__init__(in_features, out_features, bias, qconfig)
        self.init_injection()

    def forward(self, input):
        return F.linear(
            input,
            self.weight_fake_quant_inject(
                self.weight_fake_quant,
                self.weight_fake_quant(self.weight)),
            self.bias)


class RandomBETConv2D(InjectionConv2d, RandomBET):
    pass


class RandomBETLinear(InjectionLinear, RandomBET):
    pass


class BlindRowHammerAttackConv2D(InjectionConv2d, BlindRowHammerAttack):
    pass


class BlindRowHammerAttackLinear(InjectionLinear, BlindRowHammerAttack):
    pass


class RowHammerSprayAttackConv2D(InjectionConv2d, RowHammerSprayAttack):
    pass


class RowHammerSprayAttackLinear(InjectionLinear, RowHammerSprayAttack):
    pass


class RowHammerUpSprayAttackConv2D(InjectionConv2d, RowHammerUpSprayAttack):
    pass


class RowHammerUpSprayAttackLinear(InjectionLinear, RowHammerUpSprayAttack):
    pass


class BatchAblation(nn.Module):

    def __init__(self, p=0.5, q=0.3):
        super().__init__()
        self.p = p
        self.q = q

    def forward(self, input: Tensor) -> Tensor:
        if random.random() > self.q or not self.training:
            return input
        reduction_dims = tuple(d for d in range(len(input.shape)) if d != 1)
        repeat_interleaved = 1
        for d in input.shape[2:]:
            repeat_interleaved *= d
        repeat = input.shape[0]
        concise_mean = torch.mean(input, reduction_dims)
        rand = torch.rand(concise_mean.shape, device=input.device)
        concise_mean_mask = rand < self.p
        concise_value_mask = rand >= self.p
        mean = torch.repeat_interleave(concise_mean, repeat_interleaved).repeat(repeat).reshape(input.shape)
        mean_mask = torch.repeat_interleave(concise_mean_mask, repeat_interleaved).repeat(repeat).reshape(input.shape)
        value_mask = torch.repeat_interleave(concise_value_mask, repeat_interleaved).repeat(repeat).reshape(input.shape)
        return input * value_mask + mean * mean_mask


class Gaussian(nn.Module):

    def __init__(self, sigma=args.sigma, q=1):
        super().__init__()
        self.sigma = sigma
        self.q = q

    def forward(self, input: Tensor) -> Tensor:
        if random.random() > self.q or not self.training:
            return input
        mapped_sigma = float(torch.std(input) * self.sigma)
        if mapped_sigma:
            noise = torch.normal(0., mapped_sigma, input.shape, device=input.device)
            return noise + input
        else:
            return input


class MonteCarloDropout(nn.Dropout):

    def eval(self):
        if args.mc_dropout:
            return super().train()
        else:
            return super().eval()


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.features = nn.Sequential(OrderedDict({
            '0': nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            '1': nn.ReLU(inplace=True),
            '2': nn.MaxPool2d(kernel_size=3, stride=2),
            'g_1': Gaussian(),
            '3': nn.Conv2d(64, 192, kernel_size=5, padding=2),
            '4': nn.ReLU(inplace=True),
            '5': nn.MaxPool2d(kernel_size=3, stride=2),
            'g_2': Gaussian(),
            '6': nn.Conv2d(192, 384, kernel_size=3, padding=1),
            '7': nn.ReLU(inplace=True),
            'g_3': Gaussian(),
            '8': nn.Conv2d(384, 256, kernel_size=3, padding=1),
            '9': nn.ReLU(inplace=True),
            'g_4': Gaussian(),
            '10': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            '11': nn.ReLU(inplace=True),
            '12': nn.MaxPool2d(kernel_size=3, stride=2),
        }))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(OrderedDict({
            'g_5': Gaussian(),
            '0': MonteCarloDropout(p=args.dropout_p if args.first_dropout else 0),
            '1': nn.Linear(256 * 6 * 6, 4096),
            '2': nn.ReLU(inplace=True),
            # 'ba_6': BatchAblation(),
            '3': MonteCarloDropout(p=args.dropout_p),
            '4': nn.Linear(4096, 4096),
            '5': nn.ReLU(inplace=True),
            '6': MonteCarloDropout(p=args.dropout_p if args.last_dropout else 0),
            # 'ba_7': BatchAblation(),
            '7': nn.Linear(4096, num_classes),
        }))
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fill_k(self, k):
        prone_modules = [
            m for m in chain(self.features.modules(), self.classifier.modules())
            if False
               or isinstance(m, InjectionConv2d)
               or isinstance(m, InjectionLinear)
        ]
        for m in prone_modules:
            m.k = 0
        if k == 0:
            return
        # prone_modules = choices(prone_modules, [m.prob for m in prone_modules], k=1)
        for _ in range(k):
            prone = choices(prone_modules, [m.prob for m in prone_modules], k=1)[0]
            prone.k += 1


model_fp32 = AlexNet()
model_fp32.train()
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

RandomBETMapping = copy(get_qat_module_mappings())
RandomBETMapping.update({
    nn.Conv2d: RandomBETConv2D,
    nn.Linear: RandomBETLinear
})
BlindRowHammerAttackMapping = copy(get_qat_module_mappings())
BlindRowHammerAttackMapping.update({
    nn.Conv2d: BlindRowHammerAttackConv2D,
    nn.Linear: BlindRowHammerAttackLinear
})
RowHammerSprayAttackMapping = copy(get_qat_module_mappings())
RowHammerSprayAttackMapping.update({
    nn.Conv2d: RowHammerSprayAttackConv2D,
    nn.Linear: RowHammerSprayAttackLinear
})

RowHammerUpSprayAttackMapping = copy(get_qat_module_mappings())
RowHammerUpSprayAttackMapping.update({
    nn.Conv2d: RowHammerUpSprayAttackConv2D,
    nn.Linear: RowHammerUpSprayAttackLinear
})

if args.train:
    mapping = eval(args.t_mapping)
else:
    mapping = eval(args.e_mapping)

print(mapping)

model_fp32_prepared = torch.quantization.prepare_qat(model_fp32, mapping=mapping)

model_out_path = 'weights/' + serialize_params(args, exclude=('e_mapping', 'mc_dropout')) + ".pth"

if not args.train:
    assert os.path.exists(model_out_path)

load_path = model_out_path
if not os.path.exists(model_out_path):
    load_path = 'weight_backup_3_gaussian/randbet.pth'

if os.path.exists(load_path):
    if torch.cuda.is_available():
        state_dict = torch.load(load_path)
    else:
        state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    model_fp32_prepared.load_state_dict(state_dict, strict=not args.train)
    print("Checkpoint loaded from {}".format(load_path))


import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np

from misc import progress_bar

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    solver = Solver(args)
    solver.run()


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-0.1, 0.1)
            module.weight.data = w


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.evaluation_file_name = 'results/' + serialize_params(args,
                                                                  exclude_defaults={'mc_dropout': False}) + '.pkl'
        self.evaluation_file_lock = SoftFileLock(self.evaluation_file_name + '.lock')

    def log_accuracy(self, evaluation):
        print(evaluation)
        with self.evaluation_file_lock:
            try:
                with open(self.evaluation_file_name, mode='rb') as f:
                    evaluations = pickle.load(f)
            except (FileNotFoundError, pickle.UnpicklingError):
                evaluations = []
            evaluations.append(evaluation)
            with open(self.evaluation_file_name, mode='wb') as f:
                pickle.dump(evaluations, f)

    def load_data(self):
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size,
                                                        # sampler=range(self.train_batch_size),
                                                        shuffle=True
                                                        )
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size,
                                                       # sampler=range(self.test_batch_size),
                                                       shuffle=True
                                                       )

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = model_fp32_prepared.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            if args.t_mapping == 'RandomBETMapping':
                loss = loss / 2
            loss.backward()
            if args.t_mapping == 'RandomBETMapping':
                InjectionMixin.enabled = False
                output = self.model(data)
                loss = self.criterion(output, target) / 2
                loss.backward()
                InjectionMixin.enabled = True
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))
        return train_loss, train_correct / total

    def profile_grad(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            target = torch.randint(0, 10, target.shape)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))
        return train_loss, train_correct / total

    def test(self, k=0):
        print("test k = {}:".format(k))
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                # for batch_num, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.model.fill_k(k)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        clipper = WeightClipper()
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            print("\n===> epoch: {}/{}".format(epoch, args.epoch))
            print(serialize_params(args))
            if args.train:
                train_result = self.train()
                if args.weight_clip:
                    self.model.apply(clipper)
                self.scheduler.step()
                print(train_result)
            # self.profile_grad()
            test_result = self.test()
            if not args.train:
                self.log_accuracy(test_result[1])
            if test_result[1] > accuracy:
                accuracy = test_result[1]
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                if args.train:
                    self.save()


if __name__ == '__main__':
    main()

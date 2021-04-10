import os
from copy import copy
from random import choices, choice, randint, shuffle

from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.qat as nnqat
from torch.quantization import get_qat_module_mappings


class InjectionMixin:

    def init_injection(self):
        self.k = 0
        self.fault_stride = 3
        self.prob = self.weight.flatten().shape[0]
        self.grad_sum = torch.zeros(self.weight.shape)

    def weight_fake_quant_inject(self, fake_quant, weight):
        weight = self.manipulate_float(weight)
        repeat = weight.flatten().shape[0] // fake_quant.scale.flatten().shape[0]
        scale = torch.repeat_interleave(fake_quant.scale, repeat).reshape(weight.shape)
        zero_point = torch.repeat_interleave(fake_quant.zero_point, repeat).reshape(weight.shape)
        quantized = torch.clamp(
            torch.round(
                weight / scale + zero_point),
            fake_quant.quant_min,
            fake_quant.quant_max)
        quantized = self.manipulate_quantized(quantized)
        return (quantized - zero_point) * scale

    def manipulate_float(self, weight):
        return weight

    def manipulate_quantized(self, quantized):
        return quantized


class RandomBET(InjectionMixin):
    berr = 0.01

    def manipulate_float(self, weight):
        return torch.clamp(weight, -0.1, 0.1)

    def manipulate_quantized(self, signed_quantized):
        quantized = signed_quantized + 128
        mask = torch.rand(quantized.shape) > self.berr * torch.randint(0, 2, (1,)) * 8
        bit_index = torch.randint(0, 8, quantized.shape)
        bit_magnitude = 2 ** bit_index
        flip_sign = torch.masked_fill(- (torch.floor(quantized / bit_magnitude) % 2 - 0.5) * 2, mask, 0)
        additive = flip_sign * bit_magnitude
        return (quantized + additive) - 128
    
    
class RowHammerSprayAttack(InjectionMixin):
    berr = 0.007

    def manipulate_quantized(self, signed_quantized):
        quantized = signed_quantized + 128
        mask = torch.rand(quantized.shape) > self.berr * 8
        bit_index = torch.randint(0, 8, quantized.shape)
        bit_magnitude = 2 ** bit_index
        flip_sign = torch.masked_fill(- (torch.floor(quantized / bit_magnitude) % 2 - 0.5) * 2, mask, 0)
        additive = flip_sign * bit_magnitude
        return (quantized + additive) - 128


class BlindRowHammerAttack(InjectionMixin):

    def manipulate_quantized(self, signed_quantized):
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


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(AlexNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
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

model_out_path = "big_alexnet.pth"
if os.path.exists(model_out_path):
    if torch.cuda.is_available():
        state_dict = torch.load(model_out_path)
    else:
        state_dict = torch.load(model_out_path, map_location=torch.device('cpu'))
    model_fp32.load_state_dict(state_dict)
    print("Checkpoint loaded from {}".format(model_out_path))

# model_fp32_prepared = torch.quantization.prepare_qat(model_fp32, mapping=RandomBETMapping)
# model_fp32_prepared = torch.quantization.prepare_qat(model_fp32, mapping=BlindRowHammerAttackMapping)
# model_fp32_prepared = torch.quantization.prepare_qat(model_fp32, mapping=RowHammerSprayAttackMapping)
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32)
# model_fp32_prepared = model_fp32

import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np

import argparse

from misc import progress_bar

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=30, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=128, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=128, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


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
            before = self.model.state_dict()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            after = self.model.state_dict()
            pass
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
        model_out_path = "qat.pth"
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            print("\n===> epoch: %d/200" % epoch)
            train_result = self.train()
            self.scheduler.step()
            print(train_result)
            # self.profile_grad()
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()


if __name__ == '__main__':
    main()

# # Convert the observed model to a quantized model. This does several things:
# # quantizes the weights, computes and stores the scale and bias value to be
# # used with each activation tensor, fuses modules where appropriate,
# # and replaces key operators with quantized implementations.
# model_fp32_prepared.eval()
# model_int8 = torch.quantization.convert(model_fp32_prepared)
#
# # run the model, relevant calculations will happen in int8
# res = model_int8(input_fp32)

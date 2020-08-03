import sys
import time
from torch import nn
from collections import OrderedDict
import os
import numpy as np
import skimage.io
from config import cfg
from torch.utils.data import Dataset

TOTAL_BAR_LENGTH = 80
LAST_T = time.time()
BEGIN_T = LAST_T


class TestLoader(Dataset):
    def __init__(self):
        self.fileList = []
        filePath = cfg.filePath
        with open(filePath, 'r') as fp:
            lines = fp.readlines()
        fp.close()
        for line in lines:
            self.fileList.append(line.strip())

    def __getitem__(self, index):
        feature = getFeature(self.fileList[index])
        return feature, self.fileList[index]

    def __len__(self):
        return len(self.fileList)


def saveOutput(output, path):
    output = output.transpose(0,2,3,1)
    output = np.argmax(output, axis=-1)
    output[output == 0] = 10
    output[output == 1] = 15
    output[output == 2] = 25
    output[output == 3] = 35
    output[output == 4] = 45
    output[output == 5] = 55
    output[output == 6] = 65
    output[output == 7] = 75
    output = output.astype('uint8')
    for i in range(4):
        savePath = os.path.join(cfg.saveFolder, path[0], '%s.png'%((i+1)*30))
        if not os.path.exists(os.path.dirname(savePath)):
            os.makedirs(os.path.dirname(savePath))
        skimage.io.imsave(savePath, output[i])

def getFeature(item):
    feature = []
    for i in range(21):
        path = '%s/%s_%03d.png' % (cfg.dataFolder, item, i)
        # path = '%s/%s/%s_%03d.png' % (cfg.dataFolder, item, item, i)
        image = skimage.io.imread(path)
        image[image == 255] = 0
        image = image.astype('float')
        image /= 80
        image[image > 1] = 1
        feature.append(image)
    return np.expand_dims(np.asarray(feature), axis=1)


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))



def progress_bar(current, total, msg=None):
    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()  # Reset for new bar.

    current_len = int(TOTAL_BAR_LENGTH * (current + 1) / total)
    rest_len = int(TOTAL_BAR_LENGTH - current_len) - 1

    sys.stdout.write(' %d/%d' % (current + 1, total))
    sys.stdout.write(' [')
    for i in range(current_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = '  Step: %s' % format_time(step_time)
    time_used += ' | Tot: %s' % format_time(total_time)
    if msg:
        time_used += ' | ' + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# return the formatted time
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    seconds_final = int(seconds)
    seconds = seconds - seconds_final
    millis = int(seconds*1000)

    output = ''
    time_index = 1
    if days > 0:
        output += str(days) + 'D'
        time_index += 1
    if hours > 0 and time_index <= 2:
        output += str(hours) + 'h'
        time_index += 1
    if minutes > 0 and time_index <= 2:
        output += str(minutes) + 'm'
        time_index += 1
    if seconds_final > 0 and time_index <= 2:
        output += str(seconds_final) + 's'
        time_index += 1
    if millis > 0 and time_index <= 2:
        output += str(millis) + 'ms'
        time_index += 1
    if output == '':
        output = '0ms'
    return output

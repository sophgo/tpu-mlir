import os
from .base_class import *
import numpy as np

class topx(base_class):
    def init(self, args):
        self.c1 = 0
        self.c5 = 0
        self.idx = 0
        self.args = args
        self.all_labels = []
        if self.args.label_file != '':
            if os.path.exists(self.args.label_file):
                for line in open(self.args.label_file,"r").readlines():
                    line = line.strip('\n').split(' ')
                    if len(line) == 2:
                        self.all_labels.append(int(line[1]))
                    elif len(line) == 1:
                        self.all_labels.append(int(line[0]))

    def update(self, idx, outputs, labels = None):
        if len(self.all_labels) == 0 and labels is None:
            print('error, label info not exist')
            exit(1)

        self.idx = idx + 1
        softmax_probs = outputs.reshape((self.args.batch_size,-1))
        for i in range(self.args.batch_size):
            if len(self.all_labels) > 0:
                label = self.all_labels[idx-self.args.batch_size+1+i]
            elif labels is not None:
                label = labels[i]
            top5 = softmax_probs[i].squeeze().argsort()[::-1][:5]
            top1 = top5[0]
            #print(i, 'predict:', top5, 'label:', label)
            if label == top1:
                self.c1 += 1
            if label in top5:
                self.c5 += 1

    def get_result(self):
        self.print_info()
        top1 = self.c1/self.idx
        top5 = self.c5/self.idx
        return top1, top5

    def print_info(self):
        print('idx:{0}, top1:{1:.3f}, top5:{2:.3f}'.format(self.idx, self.c1/self.idx, self.c5/self.idx))

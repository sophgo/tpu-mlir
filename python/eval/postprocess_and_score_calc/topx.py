import os
from .base_class import *
import numpy as np

class topx(base_class):
    def init(self, args):
        self.c1 = 0
        self.c5 = 0
        self.idx = 0
        self.args = args
        self.all_labels_dict = {}
        self.all_labels_list = []
        if self.args.label_file != '':
            if os.path.exists(self.args.label_file):
                for line in open(self.args.label_file,"r").readlines():
                    line = [i for i in line.strip().split(' ') if len(i.strip()) > 0]
                    if len(line) == 2:
                        self.all_labels_dict[line[0]] = int(line[1])
                    else:
                        self.all_labels_list.append(int(line[0]))
        self.have_label_file = len(self.all_labels_dict) > 0 or len(self.all_labels_list) > 0


    def update(self, idx, outputs, imgs_path = None, labels = None):
        if not self.have_label_file and labels is None:
            print('error, label info not exist')
            exit(1)

        self.idx = idx + 1
        softmax_probs = outputs.reshape((self.args.batch_size,-1))
        if imgs_path is not None:
            imgs_path = imgs_path.split(',')
            assert len(imgs_path) == self.args.batch_size
        for i in range(self.args.batch_size):
            if len(self.all_labels_list) > 0:
                label = self.all_labels_list[idx-self.args.batch_size+1+i]
            elif len(self.all_labels_dict) > 0:
                for key in self.all_labels_dict:
                    if imgs_path[i].endswith(key):
                        label = self.all_labels_dict[key]
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

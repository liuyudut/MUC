#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import sys
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *

def compute_accuracy(tg_model, tg_feature_model, class_means, evalloader, scale=None, print_info=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()

    #evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                   download=False, transform=transform_test)
    #evalset.test_data = input_data.astype('uint8')
    #evalset.test_labels = input_labels
    #evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
    #    shuffle=False, num_workers=2)

    correct = 0
    correct_icarl = 0
    correct_ncm = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_feature = np.squeeze(tg_feature_model(inputs)).cpu()
            # Compute score for iCaRL
            sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()
            # Compute score for NCM
            sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
            score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()
            # print(sqd_icarl.shape, score_icarl.shape, predicted_icarl.shape, \
                  # sqd_ncm.shape, score_ncm.shape, predicted_ncm.shape)
    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(100.*correct_icarl/total))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100.*correct_ncm/total))

    cnn_acc = 100.*correct/total
    icarl_acc = 100.*correct_icarl/total
    ncm_acc = 100.*correct_ncm/total

    return [cnn_acc, icarl_acc, ncm_acc]


def compute_accuracy_CNN(tg_model, evalloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()


    #print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_WI(tg_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, side_fc=False)
            #outputs = outputs[:, start_class: end_class]
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_Version1(tg_model, evalloader, nb_cl, nclassifier, iteration):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, side_fc=True)
            #outputs = F.softmax(outputs, dim=1)
            real_classes = int(outputs.size(1)/nclassifier)
            nstep = iteration+1
            outputs_sum = torch.zeros(outputs.size(0), real_classes).to(device)
            ##
            for i in range(nstep):
                start = nb_cl*nclassifier*i
                for j in range(nclassifier):
                    end = start+nb_cl
                    outputs_sum[:, i*nb_cl:(i+1)*nb_cl] += outputs[:, start:end]
                    start = end

            # for i in range(nstep):
            #     start = nb_cl*nclassifier*i
            #     outputs_1 = F.softmax(outputs[:, start:start+nb_cl], dim=1)
            #     outputs_2 = F.softmax(outputs[:, start+nb_cl:start + 2*nb_cl], dim=1)
            #     ratio = torch.sum(torch.abs(outputs_1 - outputs_2), 1)
            #     outputs_sum[:, i*nb_cl:(i+1)*nb_cl] = outputs_1 #(outputs_1+outputs_2) * torch.unsqueeze(2.0 - ratio, 1)

            outputs_sum = F.softmax(outputs_sum, dim=1)
            _, predicted = outputs_sum.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100. * correct / total

    return cnn_acc

def compute_discrepancy(tg_model, evalloader, nb_cl, nclassifier, iteration, discrepancy):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    total = 0
    nstep = iteration + 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, side_fc=True)
            ##
            for i in range(nstep):
                start_index = nb_cl*nclassifier*i
                for iter_1 in range(nclassifier):
                    outputs_1 = outputs[:, (start_index + nb_cl * iter_1):(start_index + nb_cl * (iter_1 + 1))]
                    outputs_1 = F.softmax(outputs_1, dim=1)
                    for iter_2 in range(iter_1 + 1, nclassifier):
                        outputs_2 = outputs[:, (start_index + nb_cl * iter_2):(start_index + nb_cl * (iter_2 + 1))]
                        outputs_2 = F.softmax(outputs_2, dim=1)
                        discrepancy[targets.size(0)*batch_idx:targets.size(0)*(batch_idx+1),i] += torch.sum(torch.abs(outputs_1 - outputs_2), 1)

    return discrepancy


def compute_accuracy_Side(tg_model, evalloader, nb_cl, nclassifier, iteration, inds):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, side_fc=True)

            batch_inds = inds[batch_idx*targets.size(0):(batch_idx+1)*targets.size(0)]

            real_classes = int(outputs.size(1)/nclassifier)
            nstep = iteration+1
            outputs_sum = torch.zeros(outputs.size(0), nb_cl).to(device)
            ##

            start = nb_cl*nclassifier*batch_inds
            for j in range(nclassifier):
                end = start+nb_cl
                outputs_sum += outputs[:, start:end]
                start = end

            outputs_sum = outputs_sum/nclassifier
            outputs_sum = F.softmax(outputs_sum, dim=1)
            _, predicted = outputs_sum.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100. * correct / total

    return cnn_acc


def compute_accuracy_Step1(tg_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct1 = 0
    correct2 = 0
    correct3 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, cls_fc=True)
            # for i in range(args.num_cls):
            outputs_1 = outputs[:, :20]
            outputs_2 = outputs[:, 20:40]
            #
            outputs_1 = F.softmax(outputs_1, dim=1)
            _, predicted1 = outputs_1.max(1)
            correct1 += predicted1.eq(targets).sum().item()
            #
            outputs_2 = F.softmax(outputs_2, dim=1)
            _, predicted2 = outputs_2.max(1)
            correct2 += predicted2.eq(targets).sum().item()
            # fusion
            outputs_fusion = outputs[:, :20] + outputs[:, 20:40]
            _, predicted3 = outputs_fusion.max(1)
            correct3 += predicted3.eq(targets).sum().item()

    cnn_acc_1 = 100. * correct1 / total
    cnn_acc_2 = 100. * correct2 / total
    cnn_acc_3 = 100. * correct3 / total

    return cnn_acc_1, cnn_acc_2, cnn_acc_3

def compute_accuracy_Step2(tg_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct1 = 0
    correct2 = 0
    correct3 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, cls_fc=True)
            #outputs = F.sigmoid(outputs)
            # for i in range(args.num_cls):
            old_outputs_1 = outputs[:, :20]
            old_outputs_2 = outputs[:, 20:40]
            old_outputs = (old_outputs_1 + old_outputs_2)/2
            #
            new_outputs_1 = outputs[:, 40:60]
            new_outputs_2 = outputs[:, 60:80]
            new_outputs = (new_outputs_1 + new_outputs_2) / 2
            ##
            final_outputs = torch.cat((old_outputs_1, new_outputs_1), dim=1)
            final_outputs = F.softmax(final_outputs, dim=1)
            _, predicted1 = final_outputs.max(1)
            correct1 += predicted1.eq(targets).sum().item()

            final_outputs = torch.cat((old_outputs_2, new_outputs_2), dim=1)
            final_outputs = F.softmax(final_outputs, dim=1)
            _, predicted2 = final_outputs.max(1)
            correct2 += predicted2.eq(targets).sum().item()

            final_outputs = torch.cat((old_outputs, new_outputs), dim=1)
            final_outputs = F.softmax(final_outputs, dim=1)
            _, predicted3 = final_outputs.max(1)
            correct3 += predicted3.eq(targets).sum().item()

    cnn_acc_1 = 100. * correct1 / total
    cnn_acc_2 = 100. * correct2 / total
    cnn_acc_3 = 100. * correct3 / total

    return cnn_acc_1, cnn_acc_2, cnn_acc_3

def compute_accuracy_AIG_Cls(tg_model, cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            feats = tg_model(inputs, cls_fc=False)
            outputs = cls_model(feats)
            #outputs = outputs[:, start_class: end_class]
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_AIG_Semantic(tg_model, policy_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    correct_gates = 0
    total = 0
    temp = 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            total += targets.size(0)
            targets = targets - start_class
            inputs = inputs.to(device)
            targets = targets.to(device)

            gates, gates_cls = policy_model(inputs, temperature=temp)
            outputs = tg_model(inputs, gates)
            #outputs_sub = outputs[:, start_class: end_class]
            outputs = F.softmax(outputs, dim=1)
            gates_cls = F.softmax(gates_cls, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            _, predicted_gates = gates_cls.max(1)
            correct_gates += predicted_gates.eq(targets).sum().item()

    cnn_acc = 100.*correct/total
    cnn_acc_gates = 100. * correct_gates / total

    return cnn_acc, cnn_acc_gates

def compute_accuracy_AIG_Semantic_Cls(tg_model, cls_model, policy_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    correct_gates = 0
    total = 0
    temp = 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            total += targets.size(0)
            targets = targets - start_class
            inputs = inputs.to(device)
            targets = targets.to(device)

            gates, gates_cls = policy_model(inputs, temperature=temp)
            feats = tg_model(inputs, gates, cls_fc=False)
            outputs = cls_model(feats)
            #outputs_sub = outputs[:, start_class: end_class]
            outputs = F.softmax(outputs, dim=1)
            gates_cls = F.softmax(gates_cls, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            _, predicted_gates = gates_cls.max(1)
            correct_gates += predicted_gates.eq(targets).sum().item()

    cnn_acc = 100.*correct/total
    cnn_acc_gates = 100. * correct_gates / total

    return cnn_acc, cnn_acc_gates


def compute_accuracy_Policy_Step1(tg_model, cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class
            feats = tg_model(inputs, gates=None, cls_fc=False)
            outputs = cls_model(feats)
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs[:, start_class:end_class].max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_Policy_Step1_Gated(tg_model, policy_model, cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    temp = 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class
            new_gates = policy_model(inputs, temperature=temp)
            feats = tg_model(inputs, gates=new_gates, cls_fc=False)
            outputs = cls_model(feats)
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs[:, start_class:end_class].max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc


def compute_accuracy_Policy_Step2(tg_model, old_cls_model, new_cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class
            feats = tg_model(inputs, gates=None, cls_fc=False)
            old_logits = old_cls_model(feats)
            new_logits = new_cls_model(feats)
            logits = torch.cat((old_logits, new_logits), 1)
            logits = logits[:, start_class:end_class]
            logits = F.softmax(logits, dim=1)

            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_Policy_Step2_Gated(tg_model, policy_model, old_cls_model, new_cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    temp = 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class

            gates = policy_model(inputs, temperature=temp)
            feats = tg_model(inputs, gates=gates, cls_fc=False)

            old_logits = old_cls_model(feats)
            new_logits = new_cls_model(feats)
            logits = torch.cat((old_logits, new_logits), 1)
            logits = logits[:, start_class:end_class]
            logits = F.softmax(logits, dim=1)

            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_AIG_Original(tg_model, evalloader, start_class, end_class, gates):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class
            if gates == True:
                outputs, _ = tg_model(inputs, temperature=1, openings=gates)
            else:
                outputs = tg_model(inputs, temperature=1, openings=gates)

            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs[:, start_class:end_class].max(1)
            correct += predicted.eq(targets).sum().item()


    #print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_AIG_2(common_model, specific_model, cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class
            feats = common_model(inputs, side=False)
            feats = specific_model(feats)
            logits = cls_model(feats)
            outputs = F.softmax(logits, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()


    #print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_AIG_Step2(common_model, task1_specific_model, task2_specific_model, task1_cls_model, task2_cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            targets = targets - start_class

            feats = common_model(inputs, side=False)
            task1_feats = task1_specific_model(feats)
            task1_logits = task1_cls_model(task1_feats)
            task2_feats = task2_specific_model(feats)
            task2_logits = task2_cls_model(task2_feats)
            logits = torch.cat((task1_logits, task2_logits), 1)
            outputs = logits[:, start_class:end_class]
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()


    #print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))

    cnn_acc = 100.*correct/total

    return cnn_acc


def compute_accuracy_without_FC(tg_model, evalloader, fc_cls, pool_classifers):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    fc_cls.eval()
    if len(pool_classifers)>0:
        for old_cls in pool_classifers:
            old_cls.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            probs = fc_cls(outputs)

            if len(pool_classifers)>0:
                for old_cls in reversed(pool_classifers):
                    old_probs = old_cls(outputs)
                    probs = torch.cat((old_probs, probs), 1)

            probs = F.softmax(probs, dim=1)
            #probs = F.sigmoid(probs)
            _, predicted = probs.max(1)
            correct += predicted.eq(targets).sum().item()


    print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))

    cnn_acc = 100.*correct/total

    return cnn_acc
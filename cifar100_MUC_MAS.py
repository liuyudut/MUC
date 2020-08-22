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
import scipy.io as sio
try:
    import cPickle as pickle
except:
    import pickle
import resnet_model
import utils_pytorch
from compute_accuracy import compute_accuracy_WI
from compute_accuracy import compute_accuracy_Version1

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--dataset_dir', default='../data/cifar-100-python', type=str)
parser.add_argument('--OOD_dir', default='../data/SVHN', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--nb_cl_fg', default=20, type=int, help='the number of classes in first group')
parser.add_argument('--nb_cl', default=20, type=int, help='Classes per group')
parser.add_argument('--nb_protos', default=0, type=int, help='Number of prototypes per class at the end')
parser.add_argument('--nb_runs', default=1, type=int, help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--ckp_prefix', default='LwF_cifar100', type=str, help='Checkpoint prefix')
parser.add_argument('--epochs', default=160, type=int, help='Epochs')
parser.add_argument('--val_epoch', default=10, type=int, help='Epochs')
parser.add_argument('--resume', default='True', action='store_true', help='resume from checkpoint')
parser.add_argument('--random_seed', default=1988, type=int, help='random seed')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--side_classifier', default=3, type=int, help='multiple classifiers')
parser.add_argument('--alpha', default=0.01, type=float, help='weight for MAS')
args = parser.parse_args()

ckp_prefix = './checkpoint/MUC_MAS/step_{}_K_{}/'.format(args.nb_cl, args.side_classifier)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    t = t.to(device)
    return Variable(t, **kwargs)


class MAS(object):
    def __init__(self, model: nn.Module, evalloader, cls_id, iteration, side_fc=False):
        self.model = model
        self.dataset = evalloader
        self.params = {n: p for n, p in self.model.named_parameters() if 'fc' not in n}
        self.precision_matrices = self._diag_fisher(cls_id, iteration, side_fc=side_fc)

    def _diag_fisher(self, cls_id, iteration, side_fc=False):
        print("Training MAS model: Classifier %d"%(cls_id))
        precision_matrices = {}
        mse_criterion = nn.MSELoss()
        mse_criterion = mse_criterion.to(device)
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for batch_idx, (inputs, targets) in enumerate(self.dataset):
            inputs, targets = inputs.to(device), targets.to(device)
            #num_old_classes = args.nb_cl * iteration
            #targets = targets - num_old_classes
            if side_fc is True:
                start_index = args.side_classifier * args.nb_cl * iteration
            else:
                start_index = args.nb_cl * iteration
            self.model.zero_grad()
            outputs = self.model(inputs, side_fc=side_fc)
            i = cls_id - 1
            Target_zeros = torch.zeros_like(outputs[:, (start_index + args.nb_cl * i):(start_index + args.nb_cl * (i + 1))]).to(device)
            loss_cls = mse_criterion(outputs[:, (start_index + args.nb_cl * i):(start_index + args.nb_cl * (i + 1))], Target_zeros)
            loss_cls.backward()
            for n, p in self.model.named_parameters():
                if 'fc' not in n:
                    precision_matrices[n].data += torch.abs(p.grad.data) / len(self.dataset)
            if (batch_idx + 1) % 200 == 0:
                print(batch_idx + 1)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        save_name_side = os.path.join(ckp_prefix + 'WI/Weigtht_Importance_step_{}_K_{}_classifier_{}.pkl').format(iteration, args.side_classifier, cls_id)
        utils_pytorch.savepickle(precision_matrices, save_name_side)

        return precision_matrices


def WI_penalty(tg_model, old_params, precision_matrices):
    loss = 0
    for n, p in tg_model.named_parameters():
        if 'fc' not in n:
            ## Note that, divide by sqrt(iteration)
            #_loss = (precision_matrices[n].data.view(-1) / np.sqrt(iteration)) * (p.view(-1) - old_params[n].data.view(-1)) ** 2
            _loss = precision_matrices[n].data.view(-1) * (p.view(-1) - old_params[n].data.view(-1)) ** 2
            loss += _loss.sum()
    return loss

########################################
assert(args.nb_cl_fg % args.nb_cl == 0)
assert(args.nb_cl_fg >= args.nb_cl)
train_batch_size       = 128            # Batch size for train
test_batch_size        = 100            # Batch size for test
eval_batch_size        = 100            # Batch size for eval
base_lr                = 0.1            # Initial learning rate
lr_strat               = [120, 160, 180]      # Epochs where learning rate gets decreased
lr_factor              = 0.1            # Learning rate decrease factor
custom_weight_decay    = 5e-4            # Weight Decay
custom_momentum        = 0.9            # Momentum
epochs                 = 200
val_epoch              = 10             # evaluate the model in every val_epoch
save_epoch             = 50             # save the model in every save_epoch
np.random.seed(args.random_seed)        # Fix the random seed
print(args)
Stage1_flag = True  # Train new model and new classifier
Stage2_flag = True  # Compute weight importance
Stage3_flag = True  # Train side classifiers with Maximum Classifier Discrepancy
Stage4_flag = True  # Compute weight stability
########################################
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
])

trainset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=True,
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False,
                                       download=True, transform=transform_test)
evalset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False,
                                       download=False, transform=transform_test)

# save accuracy
top1_acc_list = np.zeros((args.nb_runs, int(args.num_classes/args.nb_cl), int(epochs/val_epoch)))

X_train_total = np.array(trainset.train_data)
Y_train_total = np.array(trainset.train_labels)
X_valid_total = np.array(testset.test_data) # test set is used as val set
Y_valid_total = np.array(testset.test_labels)

## Load unlabeled data from SVHN
svhn_data = torchvision.datasets.SVHN(root=args.OOD_dir, download=False, transform=transform_train)
svhn_num = svhn_data.data.shape[0]
svhn_data_copy = svhn_data.data
svhn_labels_copy = svhn_data.labels

# Launch the different runs
for n_run in range(args.nb_runs):
    # Select the order for the class learning
    order_name = "./checkpoint/{}_order_run_{}.pkl".format(args.dataset, n_run)
    print("Order name:{}".format(order_name))
    if os.path.exists(order_name):
        print("Loading orders")
        order = utils_pytorch.unpickle(order_name)
    else:
        print("Generating orders")
        order = np.arange(args.num_classes)
        np.random.shuffle(order)
        utils_pytorch.savepickle(order, order_name)
    order_list = list(order)
    print(order_list)

    start_iter = 0
    for iteration in range(start_iter, int(args.num_classes/args.nb_cl)):
        # Prepare the training data for the current batch of classes
        actual_cl        = order[range(iteration*args.nb_cl,(iteration+1)*args.nb_cl)]
        indices_train_subset = np.array([i in order[range(iteration*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_train_total])
        indices_test_subset  = np.array([i in order[range(0,(iteration+1)*args.nb_cl)] for i in Y_valid_total])

        ## images
        X_train          = X_train_total[indices_train_subset]
        X_valid          = X_valid_total[indices_test_subset]
        ## labels
        Y_train          = Y_train_total[indices_train_subset]
        Y_valid          = Y_valid_total[indices_test_subset]

        # Launch the training loop
        print('Batch of classes number {0} arrives ...'.format(iteration+1))
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_valid = np.array([order_list.index(i) for i in Y_valid])
        ############################################################
        trainset.train_data = X_train.astype('uint8')
        trainset.train_labels = map_Y_train
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
        testset.test_data = X_valid.astype('uint8')
        testset.test_labels = map_Y_valid
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
        print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid), max(map_Y_valid)))
        ##############################################################
        # Add the stored exemplars to the training data
        if iteration == start_iter:
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            indices_test_subset_ori = np.array([i in order[range(0, iteration*args.nb_cl)] for i in Y_valid_total])
            X_valid_ori = X_valid_total[indices_test_subset_ori]
            Y_valid_ori = Y_valid_total[indices_test_subset_ori]

        if iteration == start_iter:
            # base classes
            tg_model = resnet_cifar.resnet32(num_classes=args.nb_cl, side_classifier=args.side_classifier)
            tg_model = tg_model.to(device)
            ref_model = None
            num_old_classes = 0
        else:

            tg_model.train()
            #increment classes
            ref_model = copy.deepcopy(tg_model) 
            ref_model = ref_model.to(device)
            ref_model.eval()
            # copy old parameters
            old_params = {n: p for n, p in ref_model.named_parameters() if 'fc' not in n}

            ## new main classifier
            num_old_classes = ref_model.fc.out_features
            in_features = ref_model.fc.in_features # dim
            new_fc = nn.Linear(in_features, args.nb_cl*(iteration+1)).cuda()
            new_fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
            new_fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
            tg_model.fc = new_fc

            ## new main classifier
            num_old_classes_side = ref_model.fc_side.out_features
            in_features = ref_model.fc.in_features # dim
            new_fc_side = nn.Linear(in_features, args.side_classifier*args.nb_cl*(iteration+1)).cuda()
            new_fc_side.weight.data[:num_old_classes_side] = ref_model.fc_side.weight.data
            new_fc_side.bias.data[:num_old_classes_side] = ref_model.fc_side.bias.data
            tg_model.fc_side = new_fc_side


########### Stage 1: Train Multiple Classifiers for each iteration #################
        if Stage1_flag is True:
            print("Stage 1: Train the model for iteration {}".format(iteration))
            # Training
            update_params = list(tg_model.parameters())
            tg_optimizer = optim.SGD(update_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
            # tg_optimizer = optim.SGD(tg_params, lr=base_lr, weight_decay=custom_weight_decay)
            tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
            cls_criterion = nn.CrossEntropyLoss()
            cls_criterion.to(device)
            for epoch in range(epochs):
                temp = 1
                tg_lr_scheduler.step()
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    if args.cuda:
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                    if iteration == start_iter:
                        outputs = tg_model(inputs, side_fc=False)
                        loss_cls = cls_criterion(outputs[:, num_old_classes:(num_old_classes+args.nb_cl)], targets)
                        loss = loss_cls
                    else:
                        targets = targets - args.nb_cl * iteration
                        outputs = tg_model(inputs)
                        loss_cls = 0
                        outputs = tg_model(inputs, side_fc=False)
                        loss_cls = cls_criterion(outputs[:, num_old_classes:(num_old_classes + args.nb_cl)], targets)

                        # weight importance loss
                        loss_importance = args.alpha * WI_penalty(tg_model, old_params, weight_importance_sum)


                        # weight stability loss
                        loss_stability = args.alpha * WI_penalty(tg_model, old_params, weight_stability_sum)

                        loss = loss_cls + loss_stability + loss_importance

                    tg_optimizer.zero_grad()
                    loss.backward()
                    tg_optimizer.step()

                if iteration==start_iter:
                    print('Epoch: %d, LR: %.4f, loss_cls: %.4f' % (epoch, tg_lr_scheduler.get_lr()[0], loss_cls.item()))
                    #print(acts)
                else:
                    print('Epoch: %d, LR: %.4f, loss_cls: %.4f, loss_stability: %.4f, loss_importance: %.4f' % (
                    epoch, tg_lr_scheduler.get_lr()[0], loss_cls.item(), loss_stability.item(), loss_importance.item()))

                # evaluate the val set
                if (epoch + 1) % val_epoch == 0:
                    tg_model.eval()
                    if iteration>start_iter:
                        ## joint classifiers
                        #num_old_classes = ref_model.fc.out_features
                        tg_model.fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
                        tg_model.fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
                    print("##############################################################")
                    # Calculate validation error of model on the original classes:
                    map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
                    # print('Computing accuracy on the original batch of classes...')
                    evalset.test_data = X_valid_ori.astype('uint8')
                    evalset.test_labels = map_Y_valid_ori
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
                    acc_old = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl*(iteration+1))
                    print('Old classes accuracy: {:.2f} %'.format(acc_old))
                    ##
                    indices_test_subset_cur = np.array([i in order[range(iteration * args.nb_cl, (iteration+1) * args.nb_cl)] for i in Y_valid_total])
                    X_valid_cur = X_valid_total[indices_test_subset_cur]
                    Y_valid_cur = Y_valid_total[indices_test_subset_cur]
                    map_Y_valid_cur = np.array([order_list.index(i) for i in Y_valid_cur])
                    # print('Computing accuracy on the original batch of classes...')
                    evalset.test_data = X_valid_cur.astype('uint8')
                    evalset.test_labels = map_Y_valid_cur
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
                    acc_cur = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl*(iteration+1))
                    print('New classes accuracy: {:.2f} %'.format(acc_cur))
                    # Calculate validation error of model on the cumul of classes:
                    acc = compute_accuracy_WI(tg_model, testloader, 0, args.nb_cl*(iteration+1))
                    print('Total accuracy: {:.2f} %'.format(acc))
                    print("##############################################################")
                    tg_model.train()
                    ## record accuracy
                    top1_acc_list[n_run, iteration, int((epoch + 1)/val_epoch)-1] = np.array(acc)

                if (epoch + 1) % save_epoch == 0:
                    # # # save feature extractor
                    ckp_name = os.path.join(ckp_prefix + 'ResNet32_Model_run_{}_step_{}.pth').format(n_run,iteration)
                    torch.save(tg_model.state_dict(), ckp_name)
########### end of Stage 1


########### Stage 2: Compute weight importance for each iteration #################
        if Stage2_flag is True:
            print("Stage 2: Compute Weight Importance for iteration {}".format(iteration))
            save_name = os.path.join(ckp_prefix + 'Weight_Importance_run_{}_step_{}.pkl').format(n_run,iteration)
            # if os.path.exists(save_name):
            #     print("Loading Weight Importance Model")
            #     weight_importance_new = utils_pytorch.unpickle(save_name)
            # else:
            if 1 == 1:
                print("Training New Weight Importance Model")
                # samples from last task
                indices_sample_set = np.array([i in order[range(iteration*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_train_total])
                X_sample = X_train_total[indices_sample_set]
                Y_sample = Y_train_total[indices_sample_set]
                map_Y_sample = np.array([order_list.index(i) for i in Y_sample])
                evalset.test_data = X_sample.astype('uint8')
                evalset.test_labels = map_Y_sample
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=False, num_workers=2)
                ## compute MAS
                stage2_model = copy.deepcopy(tg_model)
                stage2_model = stage2_model.to(device)
                stage2_model.eval()
                mas_model = MAS(stage2_model, evalloader, 1, iteration, side_fc=False)
                weight_importance_new = {}
                weight_importance_sum = {}
                for n, p in copy.deepcopy(mas_model.params).items():
                    p.data.zero_()
                    weight_importance_new[n] = variable(p.data.view(-1))
                    weight_importance_sum[n] = variable(p.data.view(-1))
                for n, p in stage2_model.named_parameters():
                    if 'fc' not in n:
                        weight_importance_new[n] = mas_model.precision_matrices[n].data.view(-1)
                        weight_importance_sum[n] = mas_model.precision_matrices[n].data.view(-1)

                ## Add old Weight Importance
                if iteration > 0:
                    save_name_old = os.path.join(ckp_prefix + 'Weight_Importance_run_{}_step_{}.pkl').format(n_run, iteration-1)
                    if os.path.exists(save_name_old):
                        print("Loading Old Weight Importance Model")
                        weight_importance_old = utils_pytorch.unpickle(save_name_old)
                        for n, p in weight_importance_sum.items():
                            weight_importance_sum[n].data += weight_importance_new[n].data

                weight_importance_sum = {n: p for n, p in weight_importance_sum.items()}
                save_name = os.path.join(ckp_prefix + 'Weight_Importance_run_{}_step_{}.pkl').format(n_run, iteration)
                utils_pytorch.savepickle(weight_importance_sum, save_name)
########### end of Stage 2

########## Stage 3: Maximum Classifier Discrepancy for each iteration #################
        if Stage3_flag is True:
            print("Stage 3: Train Side Classifiers with Maximum Classifier Discrepancy for iteration {}".format(iteration))
            stage3_model = copy.deepcopy(tg_model)
            start_index = args.nb_cl * args.side_classifier * iteration
            # print("Initialize Side Classifiers with Main Classifier")
            # for i in range(args.side_classifier):
            #     stage3_model.fc_side.weight.data[(start_index + args.nb_cl * i):(start_index + args.nb_cl * (i + 1))] = stage3_model.fc.weight.data[num_old_classes:]
            #     stage3_model.fc_side.bias.data[(start_index + args.nb_cl * i):(start_index + args.nb_cl * (i + 1))] = stage3_model.fc.bias.data[num_old_classes:]
            stage3_model = stage3_model.to(device)
            stage3_model.eval()
            stage3_model.fc_side.train()
            ## fix feature extractor and main classifier
            for n, p in stage3_model.named_parameters():
                if 'fc_side' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            stage3_lr_strat = [40, 60, 70]
            stage3_epochs = 80
            stage3_params = list(stage3_model.fc_side.parameters())
            stage3_optimizer = optim.SGD(stage3_params, lr=base_lr, momentum=custom_momentum,weight_decay=custom_weight_decay)
            stage3_lr_scheduler = lr_scheduler.MultiStepLR(stage3_optimizer, milestones=stage3_lr_strat, gamma=lr_factor)
            cls_criterion = nn.CrossEntropyLoss()
            cls_criterion.to(device)
            ## Train
            for stage3_epoch in range(stage3_epochs):
                stage3_lr_scheduler.step()
                # select a subset of SVHN data
                idx = torch.randperm(svhn_num)
                svhn_data_copy = svhn_data_copy[idx]
                svhn_labels_copy = svhn_labels_copy[idx]
                svhn_data.data = svhn_data_copy[0:len(trainset.train_labels)]
                svhn_data.labels = svhn_labels_copy[0:len(trainset.train_labels)]
                svhn_loader = torch.utils.data.DataLoader(dataset=svhn_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
                for ((batch_idx, (inputs, targets)), (batch_idx_unlabel, (inputs_unlabel, targets_unlabel))) in zip(
                        enumerate(trainloader), enumerate(svhn_loader)):
                    if args.cuda:
                        inputs, targets, inputs_unlabel, targets_unlabel = inputs.to(device), targets.to(
                            device), inputs_unlabel.to(device), targets_unlabel.to(device)

                    targets = targets - args.nb_cl * iteration
                    loss_cls = 0
                    outputs = stage3_model(inputs, side_fc=True)
                    for i in range(args.side_classifier):
                        loss_cls += cls_criterion(outputs[:, (start_index + args.nb_cl * i):(start_index + args.nb_cl * (i + 1))], targets)

                    ## discrepancy loss
                    outputs_unlabel = stage3_model(inputs_unlabel, side_fc=True)
                    loss_discrepancy = 0
                    for iter_1 in range(args.side_classifier):
                        outputs_unlabel_1 = outputs_unlabel[:, (start_index + args.nb_cl * iter_1):(start_index + args.nb_cl * (iter_1 + 1))]
                        outputs_unlabel_1 = F.softmax(outputs_unlabel_1, dim=1)
                        for iter_2 in range(iter_1 + 1, args.side_classifier):
                            outputs_unlabel_2 = outputs_unlabel[:, (start_index + args.nb_cl * iter_2):(start_index + args.nb_cl * (iter_2 + 1))]
                            outputs_unlabel_2 = F.softmax(outputs_unlabel_2, dim=1)
                            #loss_discrepancy += torch.mean(F.relu(1.0 - torch.sum(torch.abs(outputs_unlabel_1 - outputs_unlabel_2), 1)))
                            loss_discrepancy += torch.mean(torch.mean(torch.abs(outputs_unlabel_1 - outputs_unlabel_2), 1))

                    loss = loss_cls - loss_discrepancy

                    stage3_optimizer.zero_grad()
                    loss.backward()
                    stage3_optimizer.step()

                print('Epoch: %d, LR: %.4f, loss_cls: %.4f, loss_discrepancy: %.4f' % (
                    stage3_epoch, stage3_lr_scheduler.get_lr()[0], loss_cls.item() / args.side_classifier, loss_discrepancy.item()))

                # evaluate the val set
                if (stage3_epoch + 1) % 10 == 0:
                    stage3_model.fc_side.eval()
                    if iteration > start_iter:
                        ## joint classifiers
                        # num_old_classes = ref_model.fc.out_features
                        stage3_model.fc_side.weight.data[:start_index] = ref_model.fc_side.weight.data
                        stage3_model.fc_side.bias.data[:start_index] = ref_model.fc_side.bias.data
                    print("##############################################################")
                    indices_test_subset_current = np.array([i in order[range(iteration * args.nb_cl, (iteration + 1) * args.nb_cl)] for i in Y_valid_total])
                    X_valid_current = X_valid_total[indices_test_subset_current]
                    Y_valid_current = Y_valid_total[indices_test_subset_current]
                    map_Y_valid_current = np.array([order_list.index(i) for i in Y_valid_current])
                    # print('Computing accuracy on the original batch of classes...')
                    evalset.test_data = X_valid_current.astype('uint8')
                    evalset.test_labels = map_Y_valid_current
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
                    acc = compute_accuracy_Version1(stage3_model, evalloader, args.nb_cl, args.side_classifier, iteration)
                    print('Maximum Classifier Discrepancy accuracy: {:.2f} %'.format(acc))
                    print("##############################################################")
                    stage3_model.fc_side.train()

                if (stage3_epoch + 1) % 40 == 0:
                    ckp_name = os.path.join(ckp_prefix + 'MCD_ResNet32_Model_run_{}_step_{}.pth').format(n_run, iteration)
                    torch.save(stage3_model.state_dict(), ckp_name)

            ## copy old and new classifiers to tg_model
            if iteration > start_iter:
                tg_model.fc_side.weight.data[:start_index] = ref_model.fc_side.weight.data
                tg_model.fc_side.bias.data[:start_index] = ref_model.fc_side.bias.data
            tg_model.fc_side.weight.data[start_index:] = stage3_model.fc_side.weight.data[start_index:]
            tg_model.fc_side.bias.data[start_index:] = stage3_model.fc_side.bias.data[start_index:]
########### end of Stage 3

########### Stage 4: Compute Weight Stability for each iteration ##################
        if Stage4_flag is True:
            print("Stage 4: Compute Weight Stability for iteration {}".format(iteration))
            # save_name = os.path.join(ckp_prefix + 'Weigtht_Stability_run_{}_step_{}_K_{}.pkl').format(n_run, iteration, args.side_classifier)
            # if os.path.exists(save_name):
            #     print("Loading Weight Stability Model")
            #     precision_matrices_new = utils_pytorch.unpickle(save_name)
            # else:
            if 1 == 1:
                print("Training New Weight Stability Model")
                # samples from last task
                indices_sample_set = np.array([i in order[range(iteration * args.nb_cl, (iteration + 1) * args.nb_cl)] for i in Y_train_total])
                X_sample = X_train_total[indices_sample_set]
                Y_sample = Y_train_total[indices_sample_set]
                map_Y_sample = np.array([order_list.index(i) for i in Y_sample])
                evalset.test_data = X_sample.astype('uint8')
                evalset.test_labels = map_Y_sample
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=False, num_workers=2)
                ## Two-classifier case
                stage4_model = copy.deepcopy(tg_model)
                stage4_model = stage4_model.to(device)
                stage4_model.eval()
                mas_model1 = MAS(stage4_model, evalloader, 1, iteration, side_fc=True)
                ##
                stage4_model = copy.deepcopy(tg_model)
                stage4_model = stage4_model.to(device)
                stage4_model.eval()
                mas_model2 = MAS(stage4_model, evalloader, 2, iteration, side_fc=True)
                ##
                stage4_model = copy.deepcopy(tg_model)
                stage4_model = stage4_model.to(device)
                stage4_model.eval()
                mas_model3 = MAS(stage4_model, evalloader, 3, iteration, side_fc=True)
                ##
                weight_stability_sum = {}
                stability_factor = {}
                for n, p in copy.deepcopy(mas_model1.params).items():
                    p.data.zero_()
                    weight_stability_sum[n] = variable(p.data.view(-1))
                    stability_factor[n] = variable(p.data.view(-1))
                print("Compute Weight Stability")
                for n, p in stage4_model.named_parameters():
                    if 'fc' not in n:
                        WI_vec = weight_importance_new[n].data.view(-1)  # 1*N
                        cls1_vec = mas_model1.precision_matrices[n].data.view(-1)  # 1*N
                        cls2_vec = mas_model2.precision_matrices[n].data.view(-1)  # 1*N
                        cls3_vec = mas_model3.precision_matrices[n].data.view(-1)  # 1*N
                        ##
                        #cls_concat_vec = torch.cat((cls1_vec, cls2_vec), dim=0)
                        diff1_vec = (cls1_vec - WI_vec) ** 2
                        diff2_vec = (cls2_vec - WI_vec) ** 2
                        diff3_vec = (cls3_vec - WI_vec) ** 2
                        diff_vec = torch.sqrt((diff1_vec + diff2_vec + diff3_vec) / args.side_classifier) # Note!!
                        std_vec = diff_vec / (WI_vec + sys.float_info.epsilon)
                        stability_vec = torch.exp(1 - std_vec)
                        ##
                        sum_vec = (cls1_vec + cls2_vec + cls3_vec) / args.side_classifier ##np.sqrt(args.side_classifier) # Note!!
                        weight_stability_sum[n] = 1.4 * sum_vec # more stable, more important
                        ##
                        stability_factor[n] = stability_vec

                ## Add old Weight Stability
                if iteration > 0:
                    save_name_old = os.path.join(ckp_prefix + 'Weigtht_Stability_run_{}_step_{}_K_{}.pkl').format(n_run, iteration-1, args.side_classifier)
                    if os.path.exists(save_name_old):
                        print("Loading Old Weight Stability Model")
                        weight_stability_old = utils_pytorch.unpickle(save_name_old)
                        for n, p in weight_stability_sum.items():
                            weight_stability_sum[n].data += weight_stability_old[n].data

                weight_stability_sum = {n: p for n, p in weight_stability_sum.items()}
                save_name = os.path.join(ckp_prefix + 'Weigtht_Stability_run_{}_step_{}_K_{}.pkl').format(n_run, iteration, args.side_classifier)
                utils_pytorch.savepickle(weight_stability_sum, save_name)
########### end of Stage 4

##################################################################
        # Final save of the results
        print("Save accuracy results for iteration {}".format(iteration))
        ckp_name = os.path.join(ckp_prefix + 'MUC_MAS_SUM_top1_acc_list_K={}_stability_1.4.mat').format(args.side_classifier)
        sio.savemat(ckp_name, {'accuracy': top1_acc_list})
##################################################################

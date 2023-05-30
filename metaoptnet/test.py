# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tqdm import tqdm
from pathlib import Path

from models.protonet_embedding import ProtoNetEmbedding
from models.R2D2_embedding import R2D2Embedding
from models.ResNet12_embedding import resnet12

from models.classification_heads import ClassificationHead

from utils import pprint, set_gpu, Timer, count_accuracy, log

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_model(config):
    # Choose the embedding network
    if config['network'] == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif config['network'] == 'R2D2':
        network = R2D2Embedding().cuda()
    elif config['network'] == 'ResNet':
        if config['dataset'] == 'miniImageNet' or config['dataset'] == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network, device_ids=[0])
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)

    # Choose the classification head
    if config['head'] == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif config['head'] == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif config['head'] == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif config['head'] == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    elif config['head'] == 'Sparse-SVM':
        cls_head = ClassificationHead(base_learner='Sparse-SVM').cuda()
    else:
        print ("Cannot recognize the classification head type")
        assert(False)

    return (network, cls_head)

def get_dataset(config):
    # Choose the embedding network
    if config['dataset'] == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif config['dataset'] == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_test = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif config['dataset'] == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif config['dataset'] == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_test = FC100(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)

    return (dataset_test, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./experiments/exp_1/best_model.pth',
                            help='path of the checkpoint file')
    parser.add_argument('--run_id', type=str, required=True,
                            help='run id in Wandb')
    parser.add_argument('--episode', type=int, default=1000,
                            help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5,
                            help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--query', type=int, default=15,
                            help='number of query examples per training class')

    opt = parser.parse_args()

    run = None  # Get run from wandb
    (dataset_test, data_loader) = get_dataset(run.config)

    # Create temporary folder
    root = Path('.') / run.id
    root.mkdir(exist_ok=True)

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode, # num of batches per epoch
    )

    set_gpu(run.config['gpu'])

    # Define the models
    (embedding_net, cls_head) = get_model(run.config)

    # Load saved model checkpoints
    saved_models = torch.load(root / 'best_model.pth')
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    cls_head.load_state_dict(saved_models['head'])
    cls_head.eval()

    # Evaluate on test set
    test_accuracies = []
    patterns = []
    for i, batch in enumerate(tqdm(dloader_test()), 1):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query

        emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
        emb_support = emb_support.reshape(1, n_support, -1)

        emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
        emb_query = emb_query.reshape(1, n_query, -1)

        if run.config['head'] == 'SVM':
            logits, sparsity = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot, maxIter=3)
        else:
            logits, sparsity = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot,
                lambda1=run.config['lambda1'], lambda2=run.config['lambda2'],
                num_steps=run.config['num_steps'], dual_reg=run.config['dual_reg']
            )

        patterns.append(sparsity.detach().cpu().numpy())

        acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())

        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)

        if i % 50 == 0:
            print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'\
                  .format(i, opt.episode, avg, ci95, acc))

    patterns = np.concatenate(patterns, axis=0)
    with open(root / 'sparsity_patterns.npy', 'wb') as f:
        np.save(f, patterns)

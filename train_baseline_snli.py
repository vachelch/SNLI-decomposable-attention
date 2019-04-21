'''
baseline model:
    standard intra-atten
    share parameters by default
'''

import logging
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchtext.data as data
import torchtext.datasets as datasets

from models.mydatasets import datasets

import time
import numpy as np
import sys
from models.baseline_snli import encoder
from models.baseline_snli import atten
import argparse


# load SST dataset
def snli(text_field, label_field,  **kargs):
    train_data, dev_data, test_data = datasets.SNLI.splits(text_field, label_field)
    # text_field.build_vocab(train_data, dev_data, test_data, vectors=args.w2v_file)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args.batch_size, 
                                                     args.batch_size, 
                                                     args.batch_size),
                                        **kargs)
    return train_iter, dev_iter, test_iter 

def train(args):

    # initialize the logger
    # create logger
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(args.log_dir + args.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # stream handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    for arg in vars(args):
        logger.info(str(arg) + ' ' + str(getattr(args, arg)))

    # load train/dev/test data
    # train data
    logger.info('loading data...')
    text_field = data.Field(lower=True, batch_first = True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter, test_iter = snli(text_field, label_field, repeat=False)

    batch_num_train = int(len(train_iter.dataset) / args.batch_size)
    train_lbl_size = len(label_field.vocab) - 1
    
    logger.info('train size # sent ' + str(len(train_iter.dataset)))
    logger.info('dev size # sent ' + str(len(dev_iter.dataset)))
    logger.info('test size # sent ' + str(len(test_iter.dataset)))

    # get input embeddings
    logger.info('loading input embeddings...')
    vocab = text_field.vocab

    best_dev = []   # (epoch, dev_acc)

    # build the model
    input_encoder = encoder(len(vocab), args.embedding_size, args.hidden_size, args.para_init)
    # input_encoder.embedding.weight.data.copy_(vocab.vectors)
    # input_encoder.embedding.weight.requires_grad = False
    inter_atten = atten(args.hidden_size, train_lbl_size, args.para_init)

    input_encoder.cuda()
    inter_atten.cuda()

    para1 = filter(lambda p: p.requires_grad, input_encoder.parameters())
    para2 = inter_atten.parameters()

    if args.optimizer == 'Adagrad':
        input_optimizer = optim.Adagrad(para1, lr=args.lr, weight_decay=args.weight_decay)
        inter_atten_optimizer = optim.Adagrad(para2, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        input_optimizer = optim.Adadelta(para1, lr=args.lr)
        inter_atten_optimizer = optim.Adadelta(para2, lr=args.lr)
    else:
        logger.info('No Optimizer.')
        sys.exit()

    criterion = nn.NLLLoss(size_average=True)
    # criterion = nn.CrossEntropyLoss()

    logger.info('start to train...')
    for k in range(args.epoch):

        total = 0.
        correct = 0.
        loss_data = 0.
        train_sents = 0.

        timer = time.time()
        for i, batch in enumerate(train_iter):
            train_src_batch, train_tgt_batch, train_lbl_batch = batch.premise, batch.hypothesis, batch.label
            train_lbl_batch.sub_(1)

            if args.cuda:
                train_src_batch, train_tgt_batch, train_lbl_batch = train_src_batch.cuda(), train_tgt_batch.cuda(), train_lbl_batch.cuda()

            train_sents += args.batch_size

            input_optimizer.zero_grad()
            inter_atten_optimizer.zero_grad()

            # initialize the optimizer
            if k == 0 and optim == 'Adagrad':
                for group in input_optimizer.param_groups:
                    for p in group['params']:
                        state = input_optimizer.state[p]
                        state['sum'] += args.Adagrad_init
                for group in inter_atten_optimizer.param_groups:
                    for p in group['params']:
                        state = inter_atten_optimizer.state[p]
                        state['sum'] += args.Adagrad_init

            train_src_linear, train_tgt_linear = input_encoder(
                train_src_batch, train_tgt_batch)
            log_prob = inter_atten(train_src_linear, train_tgt_linear)

            loss = criterion(log_prob, train_lbl_batch)

            loss.backward()

            grad_norm = 0.
            para_norm = 0.

            for m in input_encoder.modules():
                if isinstance(m, nn.Linear):
                    grad_norm += m.weight.grad.data.norm() ** 2
                    para_norm += m.weight.data.norm() ** 2
                    if m.bias is not None:
                        grad_norm += m.bias.grad.data.norm() ** 2
                        para_norm += m.bias.data.norm() ** 2

            for m in inter_atten.modules():
                if isinstance(m, nn.Linear):
                    grad_norm += m.weight.grad.data.norm() ** 2
                    para_norm += m.weight.data.norm() ** 2
                    if m.bias is not None:
                        grad_norm += m.bias.grad.data.norm() ** 2
                        para_norm += m.bias.data.norm() ** 2

            grad_norm ** 0.5
            para_norm ** 0.5

            shrinkage = args.max_grad_norm / grad_norm
            if shrinkage < 1 :
                for m in input_encoder.modules():
                    # print m
                    if isinstance(m, nn.Linear):
                        m.weight.grad.data = m.weight.grad.data * shrinkage
                for m in inter_atten.modules():
                    # print m
                    if isinstance(m, nn.Linear):
                        m.weight.grad.data = m.weight.grad.data * shrinkage
                        m.bias.grad.data = m.bias.grad.data * shrinkage

            input_optimizer.step()
            inter_atten_optimizer.step()

            _, predict = log_prob.data.max(dim=1)
            total += train_lbl_batch.data.size()[0]

            correct += torch.sum(predict == train_lbl_batch.data)

            loss_data += (loss.data * args.batch_size)  # / train_lbl_batch.data.size()[0])

            if (i + 1) % args.display_interval == 0:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, loss %.3f, para-norm %.3f, grad-norm %.3f, time %.2fs, ' %
                            (k, i + 1, batch_num_train, correct.cpu().data.numpy() / total,
                             loss_data / train_sents, para_norm, grad_norm, time.time() - timer))
                train_sents = 0.
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.
            if i == batch_num_train - 1:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, loss %.3f, para-norm %.3f, grad-norm %.3f, time %.2fs, ' %
                            (k, i + 1, batch_num_train, correct.cpu().data.numpy() / total,
                             loss_data / train_sents, para_norm, grad_norm, time.time() - timer))
                train_sents = 0.
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.           

        # evaluate
        if (k + 1) % args.dev_interval == 0:
            input_encoder.eval()
            inter_atten.eval()
            correct = 0.
            total = 0.
            for batch in dev_iter:
                dev_src_batch, dev_tgt_batch, dev_lbl_batch = batch.premise, batch.hypothesis, batch.label
                dev_lbl_batch.sub_(1)

                if args.cuda:
                    dev_src_batch, dev_tgt_batch, dev_lbl_batch = dev_src_batch.cuda(), dev_tgt_batch.cuda(), dev_lbl_batch.cuda()

                # if dev_lbl_batch.data.size(0) == 1:
                #     # simple sample batch
                #     dev_src_batch=torch.unsqueeze(dev_src_batch, 0)
                #     dev_tgt_batch=torch.unsqueeze(dev_tgt_batch, 0)

                dev_src_linear, dev_tgt_linear=input_encoder(
                    dev_src_batch, dev_tgt_batch)
                log_prob=inter_atten(dev_src_linear, dev_tgt_linear)

                _, predict=log_prob.data.max(dim=1)
                total += dev_lbl_batch.data.size()[0]
                correct += torch.sum(predict == dev_lbl_batch.data)

            dev_acc = correct.cpu().data.numpy() / total
            logger.info('dev-acc %.3f' % (dev_acc))

            if (k + 1) / args.dev_interval == 1:
                model_fname = '%s%s_epoch-%d_dev-acc-%.3f' %(args.model_path, args.log_fname.split('.')[0], k, dev_acc)
                torch.save(input_encoder.state_dict(), model_fname + '_input-encoder.pt')
                torch.save(inter_atten.state_dict(), model_fname + '_inter-atten.pt')
                best_dev.append((k, dev_acc, model_fname))
                logger.info('current best-dev:')
                for t in best_dev:
                    logger.info('\t%d %.3f' %(t[0], t[1]))
                logger.info('save model!') 
            else:
                if dev_acc > best_dev[-1][1]:
                    model_fname = '%s%s_epoch-%d_dev-acc-%.3f' %(args.model_path, args.log_fname.split('.')[0], k, dev_acc)
                    torch.save(input_encoder.state_dict(), model_fname + '_input-encoder.pt')
                    torch.save(inter_atten.state_dict(), model_fname + '_inter-atten.pt')
                    best_dev.append((k, dev_acc, model_fname))
                    logger.info('current best-dev:')
                    for t in best_dev:
                        logger.info('\t%d %.3f' %(t[0], t[1]))
                    logger.info('save model!') 

            input_encoder.train()
            inter_atten.train()

    logger.info('training end!')
    # test
    best_model_fname = best_dev[-1][2]
    input_encoder.load_state_dict(torch.load(best_model_fname + '_input-encoder.pt'))
    inter_atten.load_state_dict(torch.load(best_model_fname + '_inter-atten.pt'))

    input_encoder.eval()
    inter_atten.eval()

    correct = 0.
    total = 0.

    for batch in test_iter:
        test_src_batch, test_tgt_batch, test_lbl_batch = batch.premise, batch.hypothesis, batch.label
        test_lbl_batch.sub_(1)

        if args.cuda:
            test_src_batch, test_tgt_batch, test_lbl_batch = test_src_batch.cuda(), test_tgt_batch.cuda(), test_lbl_batch.cuda()

        test_src_linear, test_tgt_linear=input_encoder(
            test_src_batch, test_tgt_batch)
        log_prob=inter_atten(test_src_linear, test_tgt_linear)

        _, predict=log_prob.data.max(dim=1)
        total += test_lbl_batch.data.size()[0]
        correct += torch.sum(predict == test_lbl_batch.data)

    test_acc = correct.cpu().data.numpy() / total
    logger.info('test-acc %.3f' % (test_acc)) 


if __name__ == '__main__':
    parser=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--w2v_file', help='pretrained word vectors file',
                        type=str, default='glove.6B.200d')

    parser.add_argument('--log_dir', help='log file directory',
                        type=str, default='log_dir')

    parser.add_argument('--log_fname', help='log file name',
                        type=str, default='log54.log')

    parser.add_argument('--gpu_id', help='GPU device id',
                        type=int, default=0)

    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=64)

    parser.add_argument('--embedding_size', help='word embedding size',
                        type=int, default=100)

    parser.add_argument('--epoch', help='training epoch',
                        type=int, default=250)

    parser.add_argument('--dev_interval', help='interval for development',
                        type=int, default=1)

    parser.add_argument('--optimizer', help='optimizer',
                        type=str, default='Adagrad')

    parser.add_argument('--Adagrad_init', help='initial accumulating values for gradients',
                        type=float, default=0.)

    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.05)

    parser.add_argument('--hidden_size', help='hidden layer size',
                        type=int, default=100)

    parser.add_argument('--max_length', help='maximum length of training sentences,\
                        -1 means no length limit',
                        type=int, default=20)

    parser.add_argument('--display_interval', help='interval of display',
                        type=int, default=1000)

    parser.add_argument('--max_grad_norm', help='If the norm of the gradient vector exceeds this renormalize it\
                               to have the norm equal to max_grad_norm',
                        type=float, default=5)

    parser.add_argument('--para_init', help='parameter initialization gaussian',
                        type=float, default=0.01)

    parser.add_argument('--weight_decay', help='l2 regularization',
                        type=float, default=5e-5)

    parser.add_argument('--model_path', help='path of model file (not include the name suffix',
                        type=str, default='snapshots/')

    parser.add_argument('--cuda', action='store_true', default=True, help='gpu')

    args=parser.parse_args()

    args.cuda = (args.cuda) and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    # args.max_lenght = 10   # args can be set manually like this
    train(args)

else:
    pass

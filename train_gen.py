import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet,generator
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric,dot_metric,\
    dot_metric_normalize,euclidean_metric_normalize,scaling,sample,Bufferswitch,euclidean_normalize_multiscale,\
    euclidean_multiscale,dot_normalize_multiscale
from torch import nn

for i in range(5):
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--max-epoch', type=int, default=200)
        parser.add_argument('--save-epoch', type=int, default=20)
        parser.add_argument('--shot', type=int, default=5)
        parser.add_argument('--query', type=int, default=15)
        parser.add_argument('--train-way', type=int, default=5)
        parser.add_argument('--test-way', type=int, default=5)
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--bound_correct', default=True)
        parser.add_argument('--distance', type=str, default='euclidean')
        parser.add_argument('--multi',default=True)
        #parser.add_argument('--lr',default=0.0001)
        parser.add_argument('--prior_mean',default=100.0)
        parser.add_argument('--prior_var',default=1600.0)
        parser.add_argument('--log_uniform_prior',default=False)

        args = parser.parse_args()

    save_path='./gen/'+args.distance+'_5shot_lamb150_noprior'+str(i)
    #save_path='./gen/test__'+str(i)
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(save_path)

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                              args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                      num_workers=8, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                            args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                    num_workers=8, pin_memory=True)

    if args.multi==False:
        gen=generator(1600,1).cuda()
    else:
        gen=generator(1600,1600).cuda()
    model = Convnet().cuda()
    #gradient_mean=Bufferswitch()

    optimizer = torch.optim.Adam(list(model.parameters())+list(gen.parameters()), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(save_path, name + '.pth'))
    def save_gen(name):
        torch.save(gen.state_dict(),osp.join(save_path,name+'.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['mean'] = []
    trlog['max_acc']=0.0
    trlog['max_acc_val']=0.0
    trlog['variance']=[]

    timer = Timer()
    for epoch in range(1, args.max_epoch + 1):

        if epoch<=150:
            lamb=1.0-(1.0/150)*epoch
        else:
            lamb=0.0
            
        lr_scheduler.step()

        model.train()
        gen.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            query=model(data_query)
            proto = model(data_shot)

            B, mean, var = gen(torch.cat([query, proto], dim=0).mean(dim=0))

            B_no=torch.ones(B.size()).cuda()*100.0

            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            if args.bound_correct:
                if args.distance == 'cosine':
                    if args.multi == False:
                        logits, logits_scaled = dot_metric_normalize(query, proto, B)
                    else:
                        logits, logits_scaled = dot_normalize_multiscale(query, proto, B)
                else:
                    if args.multi==False:
                        logits, logits_scaled = euclidean_metric_normalize(query, proto, B)
                    else:
                        logits, logits_scaled = euclidean_normalize_multiscale(query, proto, B)
            else:
                if args.distance == 'cosine':
                    logits = dot_metric(query, proto)
                else:
                    if args.multi==False:
                        logits,logits_scaled = euclidean_multiscale(query, proto,B)
                    else:
                        logits,logits_scaled = euclidean_multiscale(query, proto,B)

            if args.distance=='euclidean':
                logits, logits_no=euclidean_normalize_multiscale(query, proto, B_no)
            else:
                logits, logits_no = dot_normalize_multiscale(query, proto, B_no)

            loss = (1-lamb)*(F.cross_entropy(logits_scaled, label,size_average=False))+lamb*F.cross_entropy(logits_no, label,size_average=False)

            acc = count_acc(logits_scaled, label)

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()
        gen.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            query = model(data_query)
            proto = model(data_shot)

            B, mean, var = gen(torch.cat([query, proto], dim=0).mean(dim=0))

            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            if args.bound_correct:
                if args.distance == 'cosine':
                    if args.multi == False:
                        logits, logits_scaled = dot_metric_normalize(query, proto, B)
                    else:
                        logits, logits_scaled = dot_normalize_multiscale(query, proto, B)
                else:
                    if args.multi == False:
                        logits, logits_scaled = euclidean_metric_normalize(query, proto, B)
                    else:
                        logits, logits_scaled = euclidean_normalize_multiscale(query, proto, B)
            else:
                if args.distance == 'cosine':
                    logits,logits_scaled = dot_metric(query, proto)
                else:
                    logits,logits_scaled = euclidean_multiscale(query, proto,B)

            if args.distance == 'euclidean':
                logits, logits_no = euclidean_normalize_multiscale(query, proto, B_no)
            else:
                logits, logits_no = dot_normalize_multiscale(query, proto, B_no)

            loss = (1-lamb)*(F.cross_entropy(logits_scaled, label,size_average=False))+lamb*F.cross_entropy(logits_no, label,size_average=False)
            acc = count_acc(logits_scaled, label)

            vl.add(loss.item())
            va.add(acc)

            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        #mean=mean.item()
        #var=var.item()

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(epoch, tl, ta))
        #print('epoch {}, mean:{}, var:{}'.format(epoch, mean,var))

        if va > trlog['max_acc_val']:
            trlog['max_acc_val'] = va
            save_model('max-acc-val')
            save_gen('max-acc-val-gen')

        if ta > trlog['max_acc']:
            trlog['max_acc'] = ta
            save_model('max-acc')
            save_gen('max-acc-gen')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)
        #trlog['mean'].append(mean)
        #trlog['variance'].append(var)

        torch.save(trlog, osp.join(save_path, 'trlog'))

        save_model('epoch-last')
        save_gen('epoch-last-gen')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))
            save_gen('epoch-{}-gen'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

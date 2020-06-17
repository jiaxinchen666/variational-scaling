import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric,dot_metric,\
    dot_metric_normalize,euclidean_metric_normalize,scaling,sample,\
    Bufferswitch,euclidean_normalize_multiscale,dot_normalize_multiscale
for run_time in range(5):
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--max-epoch', type=int, default=300)
        parser.add_argument('--save-epoch', type=int, default=20)
        parser.add_argument('--shot', type=int, default=1)
        parser.add_argument('--query', type=int, default=15)
        parser.add_argument('--train-way', type=int, default=5)
        parser.add_argument('--test-way', type=int, default=5)
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--bound_correct', default=True)
        parser.add_argument('--distance', type=str, default='cosine')
        parser.add_argument('--loss',default=True)
        parser.add_argument('--init_mean',default=100.0)
        parser.add_argument('--init_var',default=0.2)
        parser.add_argument('--lr',default=0.16)
        parser.add_argument('--prior_mean',default=1.0)
        parser.add_argument('--prior_var',default=1600.0)

        args = parser.parse_args()

    save_path='./multiscale/'+args.distance+'_shot_'+str(args.shot)+'_'+str(run_time)

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

    model = Convnet().cuda()
    gradient_mean=Bufferswitch(multiscale=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(save_path, name + '.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['mean'] = []
    trlog['max_acc']=0.0
    trlog['variance']=[]

    timer = Timer()
    mean=torch.ones([1600],requires_grad=False).cuda()*args.init_mean
    var=torch.ones([1600],requires_grad=False).cuda()*args.init_var
    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            eps = torch.normal(mean=torch.zeros([1600]), std=torch.ones([1600])).cuda()
            B = eps * var + mean

            if args.bound_correct:
                if args.distance == 'cosine':
                    logits, logits_scaled = dot_normalize_multiscale(model(data_query), proto, B)
                else:
                    logits, logits_scaled = euclidean_normalize_multiscale(model(data_query), proto, B)


            loss = F.cross_entropy(logits_scaled, label)

            prob = F.softmax(logits_scaled)

            gradient_list = [
                torch.tensor(-logits[i][label[i]] + (prob[i].unsqueeze(1) * logits[i]).sum(dim=0), requires_grad=False)
                for i in range(logits.shape[0])]

            gradient_mean.set(torch.stack(gradient_list).sum(dim=0).cuda())

            acc = count_acc(logits_scaled, label)
            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            #print(gradient_mean.get())

            #mean -= args.lr*gradient_mean.get().cuda()

            mean -= args.lr * (100*gradient_mean.get().cuda()+1.0/(args.prior_var**2)*
                              (mean-torch.ones([1600],requires_grad=False).cuda()*args.prior_mean))
            #var -= args.lr * (gradient_mean.get().cuda() * eps - 1.0 / var + var/(args.prior_var**2))

            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            eps = torch.normal(mean=torch.zeros([1600]), std=torch.ones([1600])).cuda()
            B = eps * var + mean

            if args.bound_correct:
                if args.distance == 'cosine':
                    logits, logits_scaled = dot_normalize_multiscale(model(data_query), proto, B)
                else:
                    logits, logits_scaled = euclidean_normalize_multiscale(model(data_query), proto, B)

            #logits = metric_scaling(logits, label)

            loss=F.cross_entropy(logits_scaled,label)
            acc = count_acc(logits_scaled, label)

            vl.add(loss.item())
            va.add(acc)

            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(epoch, tl, ta))
        print('epoch {}, mean:{} var:{}'.format(epoch, mean, var))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)
        trlog['mean'].append(mean.tolist())
        trlog['variance'].append(var.tolist())

        torch.save(trlog, osp.join(save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

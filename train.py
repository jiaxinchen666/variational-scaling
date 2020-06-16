import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric,dot_metric,\
    dot_metric_normalize,euclidean_metric_normalize,Bufferswitch,euclidean_normalize_multiscale

for run_time in range(5):
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
        parser.add_argument('--loss',default=True)
        parser.add_argument('--init_mean',default=100.0)
        parser.add_argument('--init_var',default=0.2)
        parser.add_argument('--lr',default=0.001)
        parser.add_argument('--prior_mean',default=1.0)
        parser.add_argument('--prior_var',default=1.0)

        args = parser.parse_args()

    save_path ='./' + args.distance+'_init_'+str(args.init_mean)\
               +'_prior_'+str(args.prior_mean)+'_'+str(run_time)

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
    gradient_mean=Bufferswitch(multiscale=False)

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
    mean=torch.tensor([args.init_mean],requires_grad=False).cuda()
    var=torch.tensor([args.init_var],requires_grad=False).cuda()
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

            eps = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0)).cuda()
            B = eps * var + mean


            if args.bound_correct:
                if args.distance == 'cosine':
                    logits, logits_scaled = dot_metric_normalize(model(data_query), proto, B)
                else:
                    logits, logits_scaled = euclidean_metric_normalize(model(data_query), proto, B)
            else:
                if args.distance == 'cosine':
                    logits = dot_metric(model(data_query), proto)
                else:
                    logits = euclidean_metric(model(data_query), proto)

            if args.bound_correct:
                loss = F.cross_entropy(logits_scaled, label)
            else:
                loss = F.cross_entropy(logits, label)

            prob = F.softmax(logits_scaled)

            gradient_list = [
                torch.tensor(-logits[i][label[i]] + sum(prob[i] * logits[i]), requires_grad=False)
                for i in range(logits.shape[0])]
            gradient_mean.set(torch.tensor(sum(gradient_list)).cuda())

            acc = count_acc(logits, label)
            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            #mean -= args.lr * (gradient_mean.get().cuda())

            mean -= args.lr * (gradient_mean.get().cuda()+1.0/(args.prior_var**2)*(mean-args.prior_mean))
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

            eps = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0)).cuda()
            B = eps * var + mean

            if args.bound_correct:
                if args.distance == 'cosine':
                    logits,logits_scaled = dot_metric_normalize(model(data_query), proto,B)
                else:
                    logits,logits_scaled = euclidean_metric_normalize(model(data_query), proto, B)
            else:
                if args.distance == 'cosine':
                    logits = dot_metric(model(data_query), proto)
                else:
                    logits = euclidean_metric(model(data_query), proto)

            if args.bound_correct:
                loss = F.cross_entropy(logits_scaled, label)
            else:
                loss = F.cross_entropy(logits, label)

            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)

            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(epoch, tl, ta))
        print('epoch {}, mean={:.4f} var={:.4f}'.format(epoch, mean[0], var[0]))


        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)
        trlog['mean'].append(mean.item())
        trlog['variance'].append(var.item())

        torch.save(trlog, osp.join(save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric,dot_metric,dot_metric_normalize\
    ,euclidean_metric_normalize,euclidean_normalize_multiscale,dot_normalize_multiscale

accuracy_list=[]
loss_list=[]
for run_time in range(5):
    path='./multiscale/cosine_1shot_init_100.0_lr_100.0_'+str(run_time)
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--load', default=path+'/max-acc.pth')
        parser.add_argument('--batch', type=int, default=10000)
        parser.add_argument('--way', type=int, default=5)
        parser.add_argument('--shot', type=int, default=1)
        parser.add_argument('--query', type=int, default=1)
        parser.add_argument('--bound', type=int, default=1)
        parser.add_argument('--bound_correct', default=True)
        parser.add_argument('--multi',default=True)
        parser.add_argument('--distance',type=str,default='cosine')
        args = parser.parse_args()
        pprint(vars(args))

        set_gpu(args.gpu)

        dataset = MiniImageNet('test')
        sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
        loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)
        model = Convnet().cuda()
        model.load_state_dict(torch.load(args.load))
        model.eval()

        ave_acc = Averager()
        ave_loss = Averager()
        ave_logits=Averager()

        for i, batch in enumerate(loader, 1):
            data, _ = [_.cuda() for _ in batch]
            k = args.way * args.shot
            data_shot, data_query = data[:k], data[k:]

            x = model(data_shot)
            x = x.reshape(args.shot, args.way, -1).mean(dim=0)
            p = x

            if args.multi:
                l=torch.load(path+'/trlog')
                B = l['mean'][list.index(l['val_acc'], max(l['val_acc']))]
                B=torch.tensor(B).cuda()

            if args.bound_correct:
                if args.distance=='cosine':
                    if args.multi:
                        logits, logits_scaled = dot_normalize_multiscale(model(data_query), p, B)
                    else:
                        logits, logits_scaled = dot_metric_normalize(model(data_query), p, B)
                else:
                    if args.multi==True:
                        logits, logits_scaled = euclidean_normalize_multiscale(model(data_query), p, B)
                    else:
                        logits,logits_scaled = euclidean_metric_normalize(model(data_query), p,B)
            else:
                if args.distance == 'cosine':
                    logits = dot_metric(model(data_query), p)
                else:
                    logits = euclidean_metric(model(data_query), p)

            label = torch.arange(args.way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            #print(torch.mean(logits_scaled))
            if args.bound_correct==True:
                loss = F.cross_entropy(logits_scaled, label)
                acc = count_acc(logits_scaled, label)
            else:
                loss = F.cross_entropy(logits, label)
                acc=count_acc(logits, label)
            ave_acc.add(acc)
            ave_loss.add(loss.item())
            x = None; p = None; logits = None

    accuracy_list.append(ave_acc.item()*100)
    loss_list.append(ave_loss.item())
    print(accuracy_list)

average_acc=sum(accuracy_list)/len(accuracy_list)
print('average_ACC:{}'.format(average_acc))
accuracy_list=[accuracy_list[i]-average_acc for i in range(len(accuracy_list))]
error=sum(np.array(accuracy_list)*np.array(accuracy_list))/(len(accuracy_list)-1.0)
print('error:{}'.format(1.96*np.sqrt(error/(len(accuracy_list)))))

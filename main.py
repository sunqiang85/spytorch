import argparse
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import click
from tqdm import tqdm
import pandas as pd
# from my project
from utils import print_dict
import utils

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="config")
    parser.add_argument('--eval', dest='eval_only', action='store_true')
    parser.add_argument('--train', dest='train_only', action='store_true')
    parser.add_argument('--test', dest='test_only', action='store_true')

    parser.add_argument('--train_split', type=str, default="train") # train|trainval
    parser.add_argument('--val_split', type=str, default="val")  # val|train
    parser.add_argument('--test_split', type=str, default="test")  # test
    parser.add_argument('--resume', action='store_true')
    return parser

def init_experiment_directory(args, config):
    exp_dir = config.exp_dir
    resume = args.resume or args.eval_only
    # create the experiment directory
    if not os.path.isdir(exp_dir):
        os.system('mkdir -p ' + exp_dir)
    else:
        if resume==False:
            if click.confirm('Exp directory already exists in {}. Erase?'
                    .format(exp_dir, default=False)):
                os.system('rm -r ' + exp_dir)
                os.system('mkdir -p ' + exp_dir)
            else:
                os._exit(1)

def train( model,  train_loader, optimizer, scheduler, lossfunc, epoch, prefix='train'):
    model.train()

    # set learning rate decay policy in epoch
    if epoch < len(config.gradual_warmup_steps) and config.schedule_method == 'warm_up':
        utils.set_lr(optimizer, config.gradual_warmup_steps[epoch])
    elif (epoch in config.lr_decay_epochs)  and config.schedule_method == 'warm_up':
        utils.decay_lr(optimizer, config.lr_decay_rate)
    utils.print_lr(optimizer, prefix, epoch)

    tq = tqdm(train_loader, desc='{} E{:03d}'.format('train', epoch), ncols=100)
    for batch_idx ,item in enumerate(tq):
        data = item['data']
        target = item['class_id'].squeeze(1)
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        result = model(data)
        output = result['prob_dist']
        loss = lossfunc(output, target)
        loss.backward()
        optimizer.step()
        # set learning rate decay policy in batch decay
        if (config.schedule_method == 'batch_decay'):
            scheduler.step()

        tq.set_postfix(loss='{:.4f}'.format(loss.item()), comp='{}'.format(batch_idx * len(data)))

def val(model, val_loader, lossfunc, epoch, topk=[1]):
    tq = tqdm(val_loader, desc='{} E{:03d}'.format('val ', epoch), ncols=100)
    model.eval()
    test_loss = 0
    correct = {k:0 for k in topk}
    maxk= max(topk)
    with torch.no_grad():
        for batch_idx, item in enumerate(tq):
            data = item['data']
            target = item['class_id'].squeeze(1)
            data, target = data.cuda(), target.cuda()
            result = model(data)
            output= result['prob_dist']
            loss = lossfunc(output, target)
            test_loss += loss.item()
            # calculate top-k
            _, pred = output.topk(maxk, 1, True, True)
            batch_correct = pred.eq(target.view(-1, 1).expand_as(pred))
            for k in topk:
                correct[k] += batch_correct[:,:k].sum().item()

    test_loss /= len(val_loader.dataset)
    result = {'epoch':epoch ,'loss':test_loss}
    for k in topk:
        keyname = "accuracy_top{}".format(k)
        result[keyname]=100. * correct[k] / len(val_loader.dataset)
        print('\nVal set: Average loss: {:.4f}, Top-{} Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, k, correct[k], len(val_loader.dataset),
            100. * correct[k] / len(val_loader.dataset)))
    return result

def run(args, config):

    assert args.train_only + args.eval_only + args.test_only <= 1
    if args.train_only + args.eval_only + args.test_only < 1:
        args.train_and_val=True
    else:
        args.train_and_val=False

    # initialiaze seeds to be able to reproduce experiment on reload
    # set mannual seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    init_experiment_directory(args, config)

    # dataset
    dataset_module = importlib.import_module(config.dataset_name)
    if args.train_and_val:
        train_loader = dataset_module.get_data_loader(config, args.train_split)
        val_loader = dataset_module.get_data_loader(config, args.val_split)
    elif args.train_only:
        train_loader = dataset_module.get_data_loader(config, args.train_split)
    elif args.eval_only:
        val_loader = dataset_module.get_data_loader(config, args.val_split)
    elif args.test_only:
        test_loader = dataset_module.get_data_loader(config, args.test_split)

    # model
    ## network
    net_module = importlib.import_module(config.model_name)
    net = net_module.Net(config)
    model = nn.DataParallel(net).cuda()

    ## criterion/loss_function
    if config.criterion_name=='nll':
        criterion=F.nll_loss
    elif config.criterion_name == 'cross_entropy':
        criterion = F.cross_entropy
    elif config.criterion_name == 'BCEWithLogitsLoss':
        criterion = F.binary_cross_entropy_with_logits

    ## metrics
    topk=config.metric_topk

    # optimizer
    if config.optim_method=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    elif config.optim_method=='adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5 ** (1 / config.lr_halflife))

    # train
    ## train init
    results = []
    start_epoch = 0
    best_accuracy_top1 = 0

    ## resume init
    if args.resume or args.eval_only or args.test_only:
        if args.train_and_val or args.train_only:
            model_path = os.path.join(config.exp_dir, 'ckpt_cur_accuracy_top1_model.pth')
            engine_path = os.path.join(config.exp_dir, 'ckpt_cur_accuracy_top1_engine.pth')
            optimizer_path = os.path.join(config.exp_dir, 'ckpt_cur_accuracy_top1_optimizer.pth')
        # eval and test based on the best model
        elif args.eval_only or args.test_only:
            model_path = os.path.join(config.exp_dir, 'ckpt_best_accuracy_top1_model.pth')
            engine_path = os.path.join(config.exp_dir, 'ckpt_best_accuracy_top1_engine.pth')
            optimizer_path = os.path.join(config.exp_dir, 'ckpt_best_accuracy_top1_optimizer.pth')
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict)
        optimizer_dict = torch.load(optimizer_path)
        optimizer.load_state_dict(optimizer_dict)
        engine_dict = torch.load(engine_path)
        start_epoch = engine_dict['epoch']
        best_accuracy_top1 = engine_dict['accuracy_top1']

    ## evaluate only
    if args.eval_only:
        ### output middle features & save figures
        if config.eval_visualize:
            for plotname in config.plot_names:
                plotmodule=importlib.import_module(plotname)
                plotmodule.save_outfeature(config, model, val_loader)
                plotmodule.save_figure(config)

        result = val(model=model, val_loader=val_loader, lossfunc=criterion, epoch=start_epoch, topk=topk)
        print(result)
        return 0

    ## test only
    if args.test_only:
        pass
        return 0

    for epoch in range(start_epoch + 1, config.nb_epochs + 1):
        train(model=model, train_loader=train_loader, optimizer=optimizer, scheduler=scheduler,lossfunc=criterion, epoch=epoch)

        # train and evaluate, save the best and current
        if args.train_and_val:
            result = val(model=model, val_loader=val_loader, lossfunc=criterion, epoch=epoch, topk=topk)
            results.append(result)

        ## save best checkpoints
        if args.train_and_val and result['accuracy_top1'] > best_accuracy_top1 and not args.eval_only:
            best_accuracy_top1 = result['accuracy_top1']
            model_path = os.path.join(config.exp_dir, 'ckpt_best_accuracy_top1_model.pth')
            engine_path = os.path.join(config.exp_dir, 'ckpt_best_accuracy_top1_engine.pth')
            optimizer_path = os.path.join(config.exp_dir, 'ckpt_best_accuracy_top1_optimizer.pth')
            engine_dict = {'epoch':epoch, 'accuracy_top1':best_accuracy_top1}
            torch.save(model.state_dict(), model_path)
            torch.save(engine_dict, engine_path)
            torch.save(optimizer.state_dict(), optimizer_path)

        ## save current checkpoints
        if args.train_and_val or args.train_only:
            model_path = os.path.join(config.exp_dir, 'ckpt_cur_accuracy_top1_model.pth')
            engine_path = os.path.join(config.exp_dir, 'ckpt_cur_accuracy_top1_engine.pth')
            optimizer_path = os.path.join(config.exp_dir, 'ckpt_cur_accuracy_top1_optimizer.pth')
            engine_dict = {'epoch': epoch, 'accuracy_top1': best_accuracy_top1}
            torch.save(model.state_dict(), model_path)
            torch.save(engine_dict, engine_path)
            torch.save(optimizer.state_dict(), optimizer_path)

        ## save the last as the best for train_only
        if args.train_only and epoch == config.nb_epochs:
            model_path = os.path.join(config.exp_dir, 'ckpt_best_accuracy_top1_model.pth')
            engine_path = os.path.join(config.exp_dir, 'ckpt_best_accuracy_top1_engine.pth')
            optimizer_path = os.path.join(config.exp_dir, 'ckpt_best_accuracy_top1_optimizer.pth')
            engine_dict = {'epoch': epoch, 'accuracy_top1': best_accuracy_top1}
            torch.save(model.state_dict(), model_path)
            torch.save(engine_dict, engine_path)
            torch.save(optimizer.state_dict(), optimizer_path)


    ## save evaluate history
    if args.train_and_val and start_epoch < config.nb_epochs:
        result_path = os.path.join(config.exp_dir, 'result.csv')
        if args.resume:
            result_df = pd.read_csv(result_path)
        else:
            result_df = pd.DataFrame()
        for r in results:
            result_df = result_df.append(pd.Series(r),ignore_index=True)
        print(result_df)
        result_df.to_csv(result_path, index=False)


if __name__ == '__main__':
    parser=get_parser()
    args = parser.parse_args()
    print('='*20)
    print(args)
    config=importlib.import_module("config."+args.conf)
    print('-'*20)
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    print_dict(config_as_dict)
    # run
    run(args, config)

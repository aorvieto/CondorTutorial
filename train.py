import argparse
import sys
import time
import torch
import wandb
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import wandb
from utils import *
from utils import print_args, str2bool, get_grad_norm_squared
from models import ResNet18, vgg11_bn


def main():
    arguments = sys.argv
    print("Command-line arguments:", arguments)

    gpu = [0]

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='test_results', type=str)
    parser.add_argument('--use_wandb', type=str2bool, nargs='?',const=True)
    parser.add_argument('--uid', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument("--lr_decay", type=str2bool, nargs='?',const=True)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--beta1', default=0, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)

    args = parser.parse_args()

    print_args(args)

    ########### Setting Up Seed ###########
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ########### Setting Up GPU ########### 
    torch.cuda.set_device(gpu[0])
    device = 'cuda'
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    print(f"Using GPU {gpu_index}: {gpu_name}")

    ########### Setting Up wandb ########### 
    if args.use_wandb:
        run=wandb.init(project=args.project,config=vars(args), dir="/wandb_tmp")
    print(vars(args))

    ########### Setup Data and Model ###########    
   
    if args.dataset=="cifar10": #only option!

        #data transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        #data
        train_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=transform_train)
        validation_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=False, transform=transform_test)
        
        #trainloader subset
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=int(args.bs), shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

        #testloader
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=int(args.bs), shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)       

        #models
        if args.model == "vgg11":
            model = torch.nn.DataParallel(vgg11_bn(),gpu).cuda()
        elif args.model == "res18":
            model = torch.nn.DataParallel(ResNet18(),gpu).cuda()
        else: 
            raise NotImplementedError("Model not defined")
        criterion = nn.CrossEntropyLoss() 
    else: 
        raise NotImplementedError("Dataset not defined")        
   
    ##### iteration counter
    iteration = 0
    total_steps = int(len(train_loader)*args.epochs)

    ########### Getting number of layers ###########      
    n_groups = 0
    dim_model = 0
    with torch.no_grad():
        for param in model.parameters():   
            n_groups = n_groups + 1
            dim_model = dim_model + torch.numel(param)
    print('Model dimension: ' + str(dim_model))
    print('Number of groups: ' + str(n_groups))
    print('Number of iterations: ' + str(total_steps))
    print(f'Steps per epoch: {len(train_loader)}')

    ########### Init of Optimizers ###########  
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = args.beta1, weight_decay= args.wd)
        if args.lr_decay:
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=int(args.epochs*0.75), gamma=0.1)
    if args.optimizer == 'adam':
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2), weight_decay= args.wd, eps = 1e-8 )
        if args.lr_decay:
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=int(args.epochs*0.75), gamma=0.1)

    df = [] #stats saved here
    filename = args.model+'_'+args.dataset+'_'+args.optimizer+'_s'+str(args.seed)+'_lr'+str(args.lr)+'_decay'+str(args.lr_decay)+'_uid'+str(args.uid)+'.csv'
    print('saving in '+str(filename))

	########### Training ###########     
    for epoch in range(args.epochs):
        start_time = time.time()

        ########### Saving stats every few epochs ###########         
        model.eval()

        #computing stats: train loss
        train_loss, correct = 0, 0
        for d in train_loader:
            data, target = d[0].to(device, non_blocking=True),d[1].to(device, non_blocking=True)
            output = model(data)
            train_loss += criterion(output, target).data.item()/len(train_loader)
            pred = output.data.max(1)[1] 
            correct += pred.eq(target.data).cpu().sum()
        accuracy_train = 100. * correct.to(torch.float32) / len(train_loader.dataset)

        #computing stats: test loss
        test_loss, correct, total = 0, 0, 0
        for d in validation_loader:
            data, target = d[0].to(device, non_blocking=True),d[1].to(device, non_blocking=True)
            output = model(data)
            test_loss += criterion(output, target).data.item()/len(validation_loader)
            pred = output.data.max(1)[1] 
            correct += pred.eq(target.data).cpu().sum()
        accuracy_test = 100. * correct.to(torch.float32) / len(validation_loader.dataset)

        #saving to wandb
        if args.use_wandb:
            wandb.log({"train_loss":train_loss, "train_acc":accuracy_train,"test_loss":test_loss,"test_acc":accuracy_test}, commit=False)

        print('Epoch {}: Train L: {:.4f}, TrainAcc: {:.2f}, Test L: {:.4f}, TestAcc: {:.2f} \n'.format(epoch, train_loss, accuracy_train, test_loss, accuracy_test))

        ###########  Training Loop ########### 
    
        model.train()
        for _, batch in enumerate(train_loader):
            opt.zero_grad() 
            model.zero_grad() 

            ###########  Backprop  ###########
            data, target = batch[0].to(device),batch[1].to(device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Getting gradient norm squared
            with torch.no_grad():
                grad_norm_squared = get_grad_norm_squared(model)
                if args.use_wandb:
                    wandb.log({"grad_norm_squared":grad_norm_squared}, commit=False)

            ###########  Optimizer update for standard methods  ########### 
            effective_lr = opt.param_groups[0]["lr"]
            if args.use_wandb:
                wandb.log({"effective_lr":effective_lr}, commit=False)
                df.append({'model': args.model, 'data': args.dataset, 'opt':args.optimizer, 'seed': args.seed, 'base_lr': args.lr, 'decay_lr': args.lr_decay, 'loss': loss.item(), 'epoch': epoch, 'lr': effective_lr})
            opt.step()
            iteration = iteration +1
            
        if args.lr_decay:
            scheduler.step() 

        epoch_time = time.time()-start_time
        if args.use_wandb:
            wandb.log({"epoch_time":epoch_time})
            pd.DataFrame(df).to_csv(os.path.join('results', filename)) #backup

            
    ########### Closing Writer ###########  
    if args.use_wandb:
        run.finish()
        wandb.finish()

    return None


if __name__ == '__main__':
    main()
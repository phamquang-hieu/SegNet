
from trainers.Trainer import Trainer
from networks.segnet import SegNet
from datasets.CamVid import CamVid
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
import albumentations as A

def main(args):
    transform = None 
    train_loader = DataLoader(CamVid(mode='train', transform=transform), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(CamVid(mode='valid', transform=transform), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(CamVid(mode='test', transform=transform), batch_size=args.batch_size, shuffle=True)
    
    model = SegNet(args.num_classes)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss = nn.BCEWithLogitsLoss() 
    
    model.cuda()
    
    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      lr_scheduler=lr_scheduler, 
                      loss = loss, 
                      args = args,
                      resume=args.resume,
                      train_loader=train_loader,
                      valid_loader=valid_loader
                    )
    print("stay-tuned")
    trainer.train()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--resume", default="")
    parser.add_argument('-n', "--num_epoch", type=int, default=120, help='number of epoch')
    parser.add_argument('-bs', "--batch_size", type=int, default=4, help='batch size')
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3, help='learning rate')
    parser.add_argument('-mmt', "--momentum", type=float, default=0.9, help='momentum')
    parser.add_argument('-n_c', "--num_classes", type=int, default=11, help='number of classes in the segmentation problem')
    
    args = parser.parse_args()
    
    main(args)

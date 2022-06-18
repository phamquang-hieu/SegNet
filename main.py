

from trainers.Trainer import Trainer
from networks.segnet import SegNet
from datasets.CamVid import CamVid
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
import albumentations as A
import random
from comet_ml import Experiment
import json

def main(args, logger):
    transform = A.Compose([
        A.OneOf([
            A.RandomSizedCrop(min_max_height=(50, 101), height=720, width=960, p=0.5),
            A.PadIfNeeded(min_height=720, min_width=960, p=0.5)
        ],p=1),
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),                 
            ], p=0.8)]
        )


    random.seed(11)
        
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
                      valid_loader=valid_loader, 
                      logger=logger
                    )
    print("stay-tuned")
    trainer.train()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--resume", default="", help='path to the directory to restore ')
    parser.add_argument('-name', "--name", type=str, default="segnet")
    parser.add_argument('-n', "--num_epoch", type=int, default=120, help='number of epoch')
    parser.add_argument('-bs', "--batch_size", type=int, default=4, help='batch size')
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3, help='learning rate')
    parser.add_argument('-mmt', "--momentum", type=float, default=0.9, help='momentum')
    parser.add_argument('-n_c', "--num_classes", type=int, default=11, help='number of classes in the segmentation problem')
    
    args = parser.parse_args()
    
    # Create an experiment with your api key
    logger = Experiment(
        api_key="zZTzevPBE5M14bjosVgWeyg3u",
        project_name="semanticsegmentation",
        workspace="phamquang-hieu",
    )
    logger.set_name(args.name)
    with open("/content/drive/MyDrive/ComputerVision/{}.json".format(args.name), "w") as f:
        json.dump(logger.get_key(), f)
    main(args, logger)

import argparse
import yaml
from model import FPClassifier
import torch
from data.augmentations import Augmentations
from data.dataset import FPDataset
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
from get_optim import get_optimizer, get_scheduler
from classifier import Classifier
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, default='train.yaml', help='training config file')
    
    args = parser.parse_args()
    
    opt = args.train_cfg
    with open(opt, 'r') as config:
        opt = yaml.safe_load(config)    
    
    model = FPClassifier(pretrained=False, num_classes=opt['num_cls'])

    train_image_dir = opt['dataset']['train']['image_path']
    train_img_size = opt['dataset']['train']['img_size']
    train_augmentations = [Augmentations(opt['dataset']['augmentations'])]
    train_batch_size = opt['dataset']['train']['batch_size']

    val_image_dir = opt['dataset']['val']['image_path']
    val_img_size = opt['dataset']['val']['img_size']
    val_batch_size = opt['dataset']['val']['batch_size']

    train_dataset = FPDataset(image_path=train_image_dir,
                              augment=True,
                              augmentations=train_augmentations,
                                  image_size=train_img_size)
    
    val_dataset = FPDataset(image_path=val_image_dir,
                                  image_size=val_img_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=opt['dataset']['train']['shuffle'],
        num_workers=opt['dataset']['train']['num_workers'] 
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=val_batch_size,
        num_workers=opt['dataset']['val']['num_workers'] 
    )
    
    logger = TensorBoardLogger(opt['logger_pth'], name="continue_train_tf_efficientnet_lite4")
    loss = nn.CrossEntropyLoss()
    
    optimizer = get_optimizer(opt, model)
    
    scheduler = get_scheduler(opt, optimizer, len(train_loader))
    
    early_stopping_callback = EarlyStopping(monitor='val accuracy', patience=100)

    checkpoint_callback = ModelCheckpoint(
        dirpath="/experiments/checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val accuracy",
        mode="max")

    classifier = Classifier(model=model, scheduler=scheduler, optimizer=optimizer, loss=loss)

    trainer = pl.Trainer(gpus=1,
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         #resume_from_checkpoint='/workspaces/FaceShapeClassifier/checkpoints/best-checkpoint.ckpt', 
                         check_val_every_n_epoch=opt['val_frequency'], 
                         max_epochs=opt['epochs'])
    
    trainer.fit(model=classifier, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)
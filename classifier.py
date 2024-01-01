import pytorch_lightning as pl
import torch.nn as nn
import torch


class Classifier(pl.LightningModule):

    def __init__(self, model, scheduler, optimizer, loss=None, save_every_epoch=False):
        super().__init__()

        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.best_accuracy = 0.0
        self.save_every_epoch = save_every_epoch
        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss


    def forward(self, x):

        return self.model(x)


    def training_step(self, train_batch, batch_idx):

        img, label = train_batch
        
        output = self.forward(img)
        
        _, preds = torch.max(output, 1)
        
        loss = self.loss(output, label)
        
        corrects = torch.sum(preds == label.data) / img.shape[0]
        
        if self.scheduler.__class__ == torch.optim.lr_scheduler.CosineAnnealingLR:
            self.scheduler.step()
            
        self.log('learning rate', self.scheduler.get_lr()[0])
        
        return {'loss': loss, 'corrects': corrects}


    def training_epoch_end(self, outputs):

        losses = [x['loss'] for x in outputs]
        corrects = [x['corrects'] for x in outputs]
        
        avg_train_loss = sum(losses) / len(losses)
        train_epoch_accuracy = sum(corrects) / len(corrects)

        self.log('train accuracy', train_epoch_accuracy)
        self.log('train loss', avg_train_loss)


    def validation_step(self, val_batch, batch_idx):

        img, label = val_batch
        
        output = self.forward(img)
        
        _, preds = torch.max(output, 1)
        
        loss = self.loss(output, label)
        
        corrects = torch.sum(preds == label.data) / img.shape[0]
        
        return {'loss': loss, 'corrects': corrects}


    def validation_epoch_end(self, outputs):
        
        losses = [x['loss'] for x in outputs]
        corrects = [x['corrects'] for x in outputs]
        
        avg_val_loss = sum(losses) / len(losses)
        val_epoch_accuracy = sum(corrects) / len(corrects)

        if val_epoch_accuracy >= self.best_accuracy:
            self.best_accuracy = val_epoch_accuracy
            
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
                }, './best.ckpt')
            
        if self.save_every_epoch:
            torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            }, './epoch_' + str(self.current_epoch) + '.ckpt')

        self.log('val loss', avg_val_loss)
        self.log('val accuracy', val_epoch_accuracy)


    def test_step(self, batch, batch_idx):

        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):

        optimizer = self.optimizer
        scheduler = self.scheduler
        
        if scheduler:
            return [optimizer], [scheduler]
        
        return [optimizer]


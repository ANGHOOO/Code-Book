# torch lightning을 이용해 모델 정의
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
import torchmetrics

class Classifier(LightningModule):
    def __init__(self, num_classes, dropout_ratio, lr=0.001):
        super().__init__()
        self.learning_rate = lr
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5), # --> [16, 28, 28]
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5), # --> [32, 24, 24]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # --> [32, 12, 12]
            nn.Dropout(dropout_ratio),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5), # --> [64, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # --> [64, 4, 4]
            nn.Dropout(dropout_ratio)
        )

        self.fc = nn.Linear(64*4*4, self.num_classes) # --> [1024, 10]
        
    def forward(self, x):
        x = self.layers(x) # --> [batch_size, 64, 4, 4]
        x = x.view(x.size(0), -1) # --> [batch_size, 1024]
        outputs = self.fc(x) # --> [batch_size, 10]

        return outputs
    
    ## configure_optimizers
    # 딥러닝 모델에 사용할 최적화 알고리즘과 학습률 스케줄러를 작성
    # 학습률 스케줄러는 작성하지 않아도 무방
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self(images)

        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self(images)
        _, preds = torch.max(outputs, dim=1)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(preds, labels)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("valid_acc", acc, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self(images)
        _, preds = torch.max(outputs, dim=1)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(preds, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        _, preds = torch.max(outputs, dim=1)

        return preds
    
# Trainer 실습
    # 학습 과정에서 조기종료와 CSV 로깅 기능을 사용

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

model = Classifier(num_classes=10, dropout_ratio=0.2)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
early_stopping = EarlyStopping(monitor='valid_loss', mode='min')
csv_logger = CSVLogger(save_dir='./csv_logger', name='test')

trainer = Trainer(
    max_epochs = 100,
    accelerator = 'auto',
    callbacks = [early_stopping, lr_monitor],
    logger = csv_logger
)

trainer.fit(model, train_dataloader, valid_dataloader)
trainer.test(model, test_dataloader)
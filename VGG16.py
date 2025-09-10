# VGG16 implemented from https://www.geeksforgeeks.org/computer-vision/vgg-16-cnn-model/

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class VGG16(nn.Module):
  def __init__(self, log_dir, output_size=8, lr=0.0001):
    super().__init__()

    self.net = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Flatten(),
      nn.Linear(25088, 4096),
      nn.ReLU(),
      nn.Dropout(0.03),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Dropout(0.03),
      nn.Linear(4096, output_size),
      nn.Softmax(1),
    )

    self.lr = lr

    self.writer = SummaryWriter(log_dir=log_dir)

  def forward(self, x):
    return self.net(x)

  @staticmethod
  def loss(y_hat, y):
    fn = nn.CrossEntropyLoss()
    return fn(y_hat, y)

  def configure_optimiser(self):
    return torch.optim.Adam(self.parameters(), self.lr)

  def save(self, save_dir, trained_epochs=0):
    save_path = (save_dir + f"/VGG16_Epoch_{int(trained_epochs)}.tar")
    torch.save(
        dict(model=self.net.state_dict(),
             learningRate=self.lr,
             epochsTrained=trained_epochs),
        save_path)
    print(f"MLP saved to {save_path} at Epoch {trained_epochs}")

  def load(self, load_dir):
    checkpoint = torch.load(load_dir, weights_only=True)
    self.model.load_state_dict(checkpoint['model'])
    self.lr = checkpoint['learningRate']
    epochs = checkpoint['epochsTrained']
    return epochs
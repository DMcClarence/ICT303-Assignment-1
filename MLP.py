# Multilayer Perceptron

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class MLP(nn.Module):
  def __init__(self, log_dir, input_size=224*224*3, output_size=8, lr=1e-4):
    super().__init__()

    self.net = nn.Sequential(
      nn.Flatten(),
      nn.Linear(input_size, 1024),
      nn.ReLU(),
      nn.Dropout(0.03),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Dropout(0.03),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(0.03),
      nn.Linear(256, output_size),
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
    save_path = (save_dir + f"/MLP_Epoch_{int(trained_epochs)}.tar")
    torch.save(
        dict(model=self.net.state_dict(),
             learning_rate=self.lr,
             epochs_trained=trained_epochs),
        save_path)
    print(f"MLP saved to {save_path} at Epoch {trained_epochs}")

  def load(self, load_dir):
    checkpoint = torch.load(load_dir, weights_only=True)
    self.model.load_state_dict(checkpoint['model'])
    self.lr = checkpoint['learning_rate']
    epochs = checkpoint['epochs_trained']
    return epochs
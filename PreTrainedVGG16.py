# PreTrainedVGG16 Class

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

class PreTrainedVGG16(nn.Module):
  def __init__(self, log_dir, device="cpu", output_size=8, n_epochs=3, lr=1e-04):
    super().__init__()
    self.model = models.vgg16(weights='IMAGENET1K_V1')
    self.model.classifier[6] = torch.nn.Linear(4096, output_size)
    self.device = device
    self.lr = lr

    self.writer = SummaryWriter(log_dir=log_dir)

    for param in self.model.parameters():
        param.requires_grad = False
    for param in self.model.classifier[6].parameters():
        param.requires_grad = True
    self.model = self.model.to(device)

  def forward(self, x):
      return self.model(x)

  @staticmethod
  def loss(y_hat, y):
    fn = nn.CrossEntropyLoss()
    return fn(y_hat, y)

  def configure_optimiser(self):
    return torch.optim.Adam(self.model.classifier[6].parameters(), self.lr)

  def save(self, save_dir, trained_epochs=0):
    save_path = (save_dir + f"/PreTrained_VGG16_Epoch_{int(trained_epochs)}.tar")
    torch.save(
        dict(model=self.model.state_dict(),
             learning_rate=self.lr,
             epochs_trained=trained_epochs),
        save_path)
    print(f"PreTrained VGG16 Model saved to {save_path} at Epoch {trained_epochs}")

  def load(self, load_dir):
    checkpoint = torch.load(load_dir, weights_only=True)
    self.model.load_state_dict(checkpoint['model'])
    self.lr = checkpoint['learning_rate']
    epochs = checkpoint['epochs_trained']
    return epochs

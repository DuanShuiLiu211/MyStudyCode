import torch
from torch import nn
from torchvision import models

model = models.resnet18()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()

# 保存处于目标周期的模型、优化器参数以及损失函数
for epoch in range(10):
    if epoch == 5:
        checkpoint_path = f"./checkpoint_epoch_{epoch}.pkl"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, checkpoint_path)
        """
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    }, checkpoint_path)
        """

        # 加载处于目标周期的模型、优化器参数以及损失函数
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss = checkpoint["loss"]

import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
  weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
torch.save(model.state_dict(), 'frcnn.pth')
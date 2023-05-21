import os

import torch

from PIL import Image
from torchvision import transforms


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
convert_tensor = transforms.ToTensor()


model = torch.load('course_model_train_0.4824_0.9429.pth')
params = []
params += [n for n, p in model.named_parameters() if 'layer' in n]
print(params)
raw_img = Image.open(f'{BASE_DIR}/panther_test.jpg')
raw_img.resize((224, 224))
test_img = convert_tensor(raw_img)
result = model(test_img.unsqueeze(0))
print(torch.argmax(result), result)

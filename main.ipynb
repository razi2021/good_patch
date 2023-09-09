import os
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import datasets
from torch.utils.data import DataLoader
import cv2
from PIL import Image
from transformers import AutoProcessor, CLIPModel

device = 'cuda'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# load data
batchsize = 100
images = []
raw_images = []
ds_classNames = ["tench", "English springer", "cassette player", "chain saw", "church",
                 "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]

ds = datasets.load_dataset("frgfm/imagenette", "320px")
raw_images = ds['train']['image']

for ibatch in tqdm(range(0, len(raw_images), batchsize)):
    inputs = processor(text=[""], images=raw_images[ibatch: ibatch + batchsize], return_tensors="pt")
    images.append(inputs['pixel_values'])

images = torch.cat(images)
print(images.shape)
ds_labels = ds['train']['label']
ds_train = datasets.Dataset.from_dict({"image": images, "label": ds_labels}).with_format("torch")


model.zero_grad()
patch = torch.rand(3, 32, 32, requires_grad=True, device=device)
optimizer = torch.optim.Adam([patch], lr=0.05, betas=(0.9, 0.999), weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss()
d_loader = torch.utils.data.DataLoader(ds_train, batch_size=100, shuffle=True)

losses = []
for epoch in range(100):
    for ib, batch in enumerate(d_loader):
        optimizer.zero_grad()
        batch['image'] = batch['image'].to(device)
        batch['label'] = batch['label'].to(device)

        targets = torch.randint(0, 49, (batch['label'].shape[0],), device=device)
        for i, trg in enumerate(targets):
            x, y = trg % 7, trg // 7
            batch['image'][i, :, y*32:(y+1)*32, x*32:(x+1)*32] += patch
        
        inputs = processor(text=ds_classNames, return_tensors="pt", padding=True).to(device)
        inputs['pixel_values'] = batch['image']
        outputs = model(**inputs, output_attentions=True) # similarity score of clip
        raw_att = outputs.vision_model_output.attentions[11]
        att_out = raw_att.mean(dim=1)[:, 0, 1:].softmax(dim=1)
        # unsupervised loss
        loss_unsup = criterion(att_out, targets)
        # supervised loss
        probs = outputs.logits_per_image.softmax(dim=-1) #.detach().cpu().numpy()
        loss_sup = criterion(probs, batch['label'])

        # overall loss
        alpha = 0.3
        loss = loss_unsup + alpha * loss_sup
        loss.backward()
        optimizer.step()
        if ib % 5 == 0:
            losses.append(loss.item())
        if ib % 20 == 0:
            print(F"{epoch}:{ib}:\t{round(loss.item(), 5)}\t| {round(torch.sqrt((patch**2).sum()).item(), 4)} |")

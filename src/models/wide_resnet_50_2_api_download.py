# Archtitechture : Wide Resnet 50 2 (meaning it is 50 layers deep - millions of params...)

import torch
import ssl

# Create unverified context - trust PyTorch
ssl._create_default_https_context = ssl._create_unverified_context

# Only WRN-50-2 and WRN-101-2 exist for download
# Visit https://github.com/pytorch/vision/blob/main/hubconf.py to see versions (may change)
model = torch.hub.load(
    repo_or_dir='pytorch/vision:v0.10.0', 
    model='wide_resnet50_2',
    pretrained=True
    )

model.eval()



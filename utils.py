# import os
# import torch
# import torchvision
# from PIL import Image
# from matplotlib import pyplot as plt
# from torch.utils.data import DataLoader


# def plot_images(images):
#     plt.figure(figsize=(32, 32))
#     plt.imshow(torch.cat([
#         torch.cat([i for i in images.cpu()], dim=-1),
#     ], dim=-2).permute(1, 2, 0).cpu())
#     plt.show()


# def save_images(images, path, **kwargs):
#     grid = torchvision.utils.make_grid(images, **kwargs)
#     ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
#     im = Image.fromarray(ndarr)
#     im.save(path)


# def get_data(args):
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
#         torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
#     return dataloader


# def setup_logging(run_name):
#     os.makedirs("models", exist_ok=True)
#     os.makedirs("results", exist_ok=True)
#     os.makedirs(os.path.join("models", run_name), exist_ok=True)
#     os.makedirs(os.path.join("results", run_name), exist_ok=True)



import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import glob

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

class SingleClassImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        
        # Get all image files directly from the directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
        for ext in image_extensions:
            self.images.extend(glob.glob(os.path.join(root_dir, ext)))
        
        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Return 0 as dummy label

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use custom dataset for single-class images
    dataset = SingleClassImageDataset(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

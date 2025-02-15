import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np
from glob import glob

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Directories
CLOUDY_DIR = r'C:\\Users\\LENOVO\\Documents\\cloud removal\\FULL DATASET\\cloud'
CLOUD_FREE_DIR = r'C:\\Users\\LENOVO\\Documents\\cloud removal\\FULL DATASET\\cloudfree'
OUTPUT_DIR = r'C:\\Users\\LENOVO\\Documents\\cloud removal\\newres'
CHECKPOINT_DIR = r'C:\\Users\\LENOVO\\Documents\\cloud removal\\checkpoint'

# Create output and checkpoint directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 3
BATCH_SIZE = 4  # Increased batch size
LAMBDA_CYCLE = 5.0  # Adjusted lambda cycle
LAMBDA_IDENTITY = 0.5  # Adjusted lambda identity
EPOCHS = 200
LR = 0.0001  # Smaller learning rate
BETA1 = 0.5
BETA2 = 0.999
VALIDATION_SPLIT = 0.2

# Custom Dataset
class SatelliteDataset(Dataset):
    def __init__(self, cloudy_dir, cloud_free_dir):
        self.cloudy_paths = sorted(glob(os.path.join(cloudy_dir, '*.jpg')) + glob(os.path.join(cloudy_dir, '*.jpeg')))
        self.cloud_free_paths = sorted(glob(os.path.join(cloud_free_dir, '*.jpg')) + glob(os.path.join(cloud_free_dir, '*.jpeg')))
        if len(self.cloudy_paths) == 0 or len(self.cloud_free_paths) == 0:
            raise FileNotFoundError(f"No .jpg or .jpeg images found in directories. Check {cloudy_dir} and {cloud_free_dir}")
        print(f"Found {len(self.cloudy_paths)} cloudy images and {len(self.cloud_free_paths)} cloud-free images")
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return min(len(self.cloudy_paths), len(self.cloud_free_paths))
    
    def __getitem__(self, idx):
        cloudy_img = Image.open(self.cloudy_paths[idx]).convert('RGB')
        cloud_free_img = Image.open(self.cloud_free_paths[idx]).convert('RGB')
        cloudy_img = self.transform(cloudy_img)
        cloud_free_img = self.transform(cloud_free_img)
        return {'cloudy': cloudy_img, 'cloud_free': cloud_free_img}

# ResNet block
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features)
        )
    
    def forward(self, x):
        return x + self.block(x)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(CHANNELS, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Downsampling
        self.down_sampling = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(9)]  # 9 residual blocks
        )
        # Upsampling
        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Output convolution
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, CHANNELS, 7),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.down_sampling(x)
        x = self.res_blocks(x)
        x = self.up_sampling(x)
        return self.output(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(CHANNELS, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

def init_weights(model):
    """Initialize network weights using Xavier initialization"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

def main():
    print("Initializing models...")
    # Initialize models and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Initialize generators and discriminators
    G_cloudy2clear = Generator().to(device)
    G_clear2cloudy = Generator().to(device)
    D_cloudy = Discriminator().to(device)
    D_clear = Discriminator().to(device)
    # Initialize weights
    init_weights(G_cloudy2clear)
    init_weights(G_clear2cloudy)
    init_weights(D_cloudy)
    init_weights(D_clear)
    # Debug prints to verify parameters
    print("Generator cloudy2clear parameters:", sum(p.numel() for p in G_cloudy2clear.parameters()))
    print("Generator clear2cloudy parameters:", sum(p.numel() for p in G_clear2cloudy.parameters()))
    print("Discriminator cloudy parameters:", sum(p.numel() for p in D_cloudy.parameters()))
    print("Discriminator clear parameters:", sum(p.numel() for p in D_clear.parameters()))
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    # Optimizers
    generator_params = list(G_cloudy2clear.parameters()) + list(G_clear2cloudy.parameters())
    if len(generator_params) == 0:
        raise ValueError("No parameters found in generators!")
    optimizer_G = optim.Adam(generator_params, lr=LR, betas=(BETA1, BETA2))
    optimizer_D_cloudy = optim.Adam(D_cloudy.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D_clear = optim.Adam(D_clear.parameters(), lr=LR, betas=(BETA1, BETA2))
    # Learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
    lr_scheduler_D_cloudy = torch.optim.lr_scheduler.StepLR(optimizer_D_cloudy, step_size=50, gamma=0.5)
    lr_scheduler_D_clear = torch.optim.lr_scheduler.StepLR(optimizer_D_clear, step_size=50, gamma=0.5)
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = SatelliteDataset(CLOUDY_DIR, CLOUD_FREE_DIR)
    dataset_size = len(dataset)
    train_size = int((1 - VALIDATION_SPLIT) * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Train dataset size: {train_size} images")
    print(f"Validation dataset size: {val_size} images")
    
    def save_images(epoch, batch_idx, real_cloudy, real_clear, fake_clear, prefix='train'):
        """Save images during training"""
        images = torch.cat([real_cloudy, real_clear, fake_clear], dim=3)
        save_path = os.path.join(OUTPUT_DIR, f'{prefix}_epoch_{epoch}_batch{batch_idx}.png')
        torchvision.utils.save_image(images, save_path, normalize=True)
    
    print("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        try:
            G_loss_train = 0.0
            D_loss_train = 0.0
            G_loss_val = 0.0
            D_loss_val = 0.0
            
            # Training phase
            G_cloudy2clear.train()
            G_clear2cloudy.train()
            D_cloudy.train()
            D_clear.train()
            
            for i, batch in enumerate(train_dataloader):
                real_cloudy = batch['cloudy'].to(device)
                real_clear = batch['cloud_free'].to(device)
                
                # Generate fake images
                fake_clear = G_cloudy2clear(real_cloudy)
                fake_cloudy = G_clear2cloudy(real_clear)
                
                # Train Generators
                optimizer_G.zero_grad()
                # Identity loss
                loss_id_clear = criterion_identity(G_cloudy2clear(real_clear), real_clear)
                loss_id_cloudy = criterion_identity(G_clear2cloudy(real_cloudy), real_cloudy)
                loss_identity = (loss_id_clear + loss_id_cloudy) * LAMBDA_IDENTITY
                # GAN loss
                pred_fake_clear = D_clear(fake_clear)
                loss_GAN_cloudy2clear = criterion_GAN(pred_fake_clear, torch.ones_like(pred_fake_clear))
                pred_fake_cloudy = D_cloudy(fake_cloudy)
                loss_GAN_clear2cloudy = criterion_GAN(pred_fake_cloudy, torch.ones_like(pred_fake_cloudy))
                # Cycle loss
                recovered_cloudy = G_clear2cloudy(fake_clear)
                loss_cycle_cloudy = criterion_cycle(recovered_cloudy, real_cloudy)
                recovered_clear = G_cloudy2clear(fake_cloudy)
                loss_cycle_clear = criterion_cycle(recovered_clear, real_clear)
                loss_cycle = (loss_cycle_cloudy + loss_cycle_clear) * LAMBDA_CYCLE
                # Total generator loss
                loss_G = loss_GAN_cloudy2clear + loss_GAN_clear2cloudy + loss_cycle + loss_identity
                loss_G.backward()
                optimizer_G.step()
                
                # Train Discriminator Clear
                optimizer_D_clear.zero_grad()
                pred_real_clear = D_clear(real_clear)
                loss_D_real = criterion_GAN(pred_real_clear, torch.ones_like(pred_real_clear))
                pred_fake_clear = D_clear(fake_clear.detach())
                loss_D_fake = criterion_GAN(pred_fake_clear, torch.zeros_like(pred_fake_clear))
                loss_D_clear = (loss_D_real + loss_D_fake) * 0.5
                loss_D_clear.backward()
                optimizer_D_clear.step()
                
                # Train Discriminator Cloudy
                optimizer_D_cloudy.zero_grad()
                pred_real_cloudy = D_cloudy(real_cloudy)
                loss_D_real = criterion_GAN(pred_real_cloudy, torch.ones_like(pred_real_cloudy))
                pred_fake_cloudy = D_cloudy(fake_cloudy.detach())
                loss_D_fake = criterion_GAN(pred_fake_cloudy, torch.zeros_like(pred_fake_cloudy))
                loss_D_cloudy = (loss_D_real + loss_D_fake) * 0.5
                loss_D_cloudy.backward()
                optimizer_D_cloudy.step()
                
                # Accumulate training losses
                G_loss_train += loss_G.item()
                D_loss_train += (loss_D_clear + loss_D_cloudy).item()
                
                # Print progress
                if i % 100 == 0:
                    print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(train_dataloader)}] "
                          f"[D loss: {(loss_D_clear + loss_D_cloudy).item():.4f}] "
                          f"[G loss: {loss_G.item():.4f}]")
                
                # Save sample images
                save_images(epoch, i, real_cloudy, real_clear, fake_clear, prefix='train')
            
            # Validation phase
            G_cloudy2clear.eval()
            G_clear2cloudy.eval()
            D_cloudy.eval()
            D_clear.eval()
            
            with torch.no_grad():
                for i, batch in enumerate(val_dataloader):
                    real_cloudy = batch['cloudy'].to(device)
                    real_clear = batch['cloud_free'].to(device)
                    
                    # Generate fake images
                    fake_clear = G_cloudy2clear(real_cloudy)
                    fake_cloudy = G_clear2cloudy(real_clear)
                    
                    # Calculate validation losses
                    loss_id_clear = criterion_identity(G_cloudy2clear(real_clear), real_clear)
                    loss_id_cloudy = criterion_identity(G_clear2cloudy(real_cloudy), real_cloudy)
                    loss_identity = (loss_id_clear + loss_id_cloudy) * LAMBDA_IDENTITY
                    
                    pred_fake_clear = D_clear(fake_clear)
                    loss_GAN_cloudy2clear = criterion_GAN(pred_fake_clear, torch.ones_like(pred_fake_clear))
                    pred_fake_cloudy = D_cloudy(fake_cloudy)
                    loss_GAN_clear2cloudy = criterion_GAN(pred_fake_cloudy, torch.ones_like(pred_fake_cloudy))
                    
                    recovered_cloudy = G_clear2cloudy(fake_clear)
                    loss_cycle_cloudy = criterion_cycle(recovered_cloudy, real_cloudy)
                    recovered_clear = G_cloudy2clear(fake_cloudy)
                    loss_cycle_clear = criterion_cycle(recovered_clear, real_clear)
                    loss_cycle = (loss_cycle_cloudy + loss_cycle_clear) * LAMBDA_CYCLE
                    
                    loss_G = loss_GAN_cloudy2clear + loss_GAN_clear2cloudy + loss_cycle + loss_identity
                    
                    pred_real_clear = D_clear(real_clear)
                    loss_D_real = criterion_GAN(pred_real_clear, torch.ones_like(pred_real_clear))
                    pred_fake_clear = D_clear(fake_clear)
                    loss_D_fake = criterion_GAN(pred_fake_clear, torch.zeros_like(pred_fake_clear))
                    loss_D_clear = (loss_D_real + loss_D_fake) * 0.5
                    
                    pred_real_cloudy = D_cloudy(real_cloudy)
                    loss_D_real = criterion_GAN(pred_real_cloudy, torch.ones_like(pred_real_cloudy))
                    pred_fake_cloudy = D_cloudy(fake_cloudy)
                    loss_D_fake = criterion_GAN(pred_fake_cloudy, torch.zeros_like(pred_fake_cloudy))
                    loss_D_cloudy = (loss_D_real + loss_D_fake) * 0.5
                    
                    # Accumulate validation losses
                    G_loss_val += loss_G.item()
                    D_loss_val += (loss_D_clear + loss_D_cloudy).item()
                    
                    # Save sample images
                    save_images(epoch, i, real_cloudy, real_clear, fake_clear, prefix='val')
            
            # Average losses
            G_loss_train /= len(train_dataloader)
            D_loss_train /= len(train_dataloader)
            G_loss_val /= len(val_dataloader)
            D_loss_val /= len(val_dataloader)
            
            print(f"[Epoch {epoch}/{EPOCHS}] "
                  f"[Train D loss: {D_loss_train:.4f}] "
                  f"[Train G loss: {G_loss_train:.4f}] "
                  f"[Val D loss: {D_loss_val:.4f}] "
                  f"[Val G loss: {G_loss_val:.4f}]")
            
            # Early stopping based on validation loss
            if D_loss_val < best_val_loss:
                best_val_loss = D_loss_val
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_checkpoint.pth')
                torch.save({
                    'G_cloudy2clear': G_cloudy2clear.state_dict(),
                    'G_clear2cloudy': G_clear2cloudy.state_dict(),
                    'D_cloudy': D_cloudy.state_dict(),
                    'D_clear': D_clear.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D_cloudy': optimizer_D_cloudy.state_dict(),
                    'optimizer_D_clear': optimizer_D_clear.state_dict(),
                    'epoch': epoch
                }, checkpoint_path)
                print(f"Saved best checkpoint to {checkpoint_path}")
            
            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_cloudy.step()
            lr_scheduler_D_clear.step()
            
            # Save models periodically
            if (epoch + 1) % 50 == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'G_cloudy2clear': G_cloudy2clear.state_dict(),
                    'G_clear2cloudy': G_clear2cloudy.state_dict(),
                    'D_cloudy': D_cloudy.state_dict(),
                    'D_clear': D_clear.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D_cloudy': optimizer_D_cloudy.state_dict(),
                    'optimizer_D_clear': optimizer_D_clear.state_dict(),
                    'epoch': epoch
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        except Exception as e:
            print(f"An error occurred in epoch {epoch}: {e}")
            break
    
    print("Training finished!")

if __name__ == '__main__':
    main()
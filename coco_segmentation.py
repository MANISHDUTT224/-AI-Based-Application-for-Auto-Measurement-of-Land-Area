import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

class Config:
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 4  # Updated for multiple classes
    CLASS_NAMES = ['residential', 'golf_course', 'tennis_court', 'beach']
    
    # Paths
    ROOT_DIR = r"C:\\Users\\LENOVO\\Documents\\cloud removal\\segmentation dataset\\train"
    ANNOTATION_FILE = r"C:\\Users\\LENOVO\\Documents\\cloud removal\\segmentation dataset\\train\\_annotations.coco.json"
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):  # Fixed method name
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):  # Fixed method name
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv4(x)
        
        return self.outc(x)

class LandUseDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):  # Fixed method name
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        
        self.image_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), 
                            interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self):  # Fixed method name
        return len(self.ids)
    
    def __getitem__(self, idx):  # Fixed method name
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)
        
        # Create multi-class mask
        mask = np.zeros((Config.NUM_CLASSES, img_info['height'], img_info['width']))
        anns_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(anns_ids)
        
        for ann in anns:
            category_id = ann['category_id']
            if category_id < Config.NUM_CLASSES:  # Ensure valid category
                curr_mask = self.coco.annToMask(ann)
                mask[category_id] = np.maximum(mask[category_id], curr_mask)
        
        mask = torch.from_numpy(mask)
        mask = F.interpolate(mask.unsqueeze(0), size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), 
                           mode='nearest').squeeze(0)
        
        return image, mask

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)

def analyze_image(model, image_path, device=Config.DEVICE):
    """Analyze a single image and return segmentation results with properties"""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        output = F.softmax(output, dim=1)
        pred = output.argmax(dim=1)
    
    # Calculate properties for each class
    results = {
        "image_path": image_path,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "class_distribution": {},
        "confidence_scores": {}
    }
    
    # Calculate area and confidence for each class
    for i, class_name in enumerate(Config.CLASS_NAMES):
        mask = (pred == i).float()
        area_percentage = (mask.sum() / (Config.IMAGE_SIZE * Config.IMAGE_SIZE) * 100).item()
        confidence = output[:, i].mean().item() * 100
        
        results["class_distribution"][class_name] = {
            "area_percentage": round(area_percentage, 2),
            "pixels": int(mask.sum().item())
        }
        results["confidence_scores"][class_name] = round(confidence, 2)
    
    # Save visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(transforms.ToPILImage()(image_tensor.squeeze(0).cpu()))
    plt.title("Original Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred.squeeze().cpu().numpy())
    plt.title("Segmentation Map")
    
    plt.subplot(1, 3, 3)
    plt.bar(Config.CLASS_NAMES, 
           [results["class_distribution"][c]["area_percentage"] for c in Config.CLASS_NAMES])
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save visualization
    plt.savefig(os.path.join(Config.RESULTS_DIR, f"{base_filename}analysis{timestamp}.png"))
    plt.close()
    
    # Save JSON results
    json_path = os.path.join(Config.RESULTS_DIR, f"{base_filename}results{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    # Training
    print(f"Using device: {Config.DEVICE}")
    
    dataset = LandUseDataset(
        root_dir=Config.ROOT_DIR,
        annotation_file=Config.ANNOTATION_FILE
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    model = UNet(n_channels=3, n_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    train_losses = []
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_loss = train_model(model, train_loader, criterion, optimizer, Config.DEVICE)
        train_losses.append(train_loss)
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'unet_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(Config.CHECKPOINT_DIR, 'unet_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': train_losses[-1],
    }, final_model_path)
    
    return model

def inference(image_path, model_path=None):
    """Run inference on a single image"""
    # Load model
    model = UNet(n_channels=3, n_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Analyze image
    results = analyze_image(model, image_path)
    print("\nAnalysis Results:")
    print(json.dumps(results, indent=4))
    print(f"\nResults saved in {Config.RESULTS_DIR}")
    
    return results

if __name__ == '__main__':
    # Training
    model = main()
    
    # Example inference
    # Uncomment and modify path to run inference on a specific image
    # image_path = "path/to/your/test/image.jpg"
    # results = inference(image_path)
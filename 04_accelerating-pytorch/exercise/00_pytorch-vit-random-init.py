import time

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torchvision import transforms
from torchvision.models import vit_b_16
from torchvision.models import ViT_B_16_Weights
from watermark import watermark

from local_utilities import get_dataloaders_cifar10


def train(num_epochs, model, optimizer, train_loader, val_loader, device):

    for epoch in range(num_epochs):
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            model.train()

            features = features.to(device)
            targets = targets.to(device)
            
            ### FORWARD AND BACK PROP   
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if not batch_idx % 300:
                print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss:.4f}")

            model.eval()
            with torch.no_grad():
                predicted_labels = torch.argmax(logits, 1)
                train_acc.update(predicted_labels, targets)

        ### MORE LOGGING
        model.eval()
        with torch.no_grad():
            val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

            for (features, targets) in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                predicted_labels = torch.argmax(outputs, 1)
                val_acc.update(predicted_labels, targets)

            print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train acc.: {train_acc.compute()*100:.2f}% | Val acc.: {val_acc.compute()*100:.2f}%")
            train_acc.reset(), val_acc.reset()


if __name__ == "__main__":

    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    L.seed_everything(123)

    ##########################
    ### 1 Loading the Dataset
    ##########################
    train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           #transforms.RandomCrop((224, 224)),
                                           transforms.ToTensor()])
    
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          #transforms.CenterCrop((224, 224)),
                                          transforms.ToTensor()])
    
    train_loader, val_loader, test_loader = get_dataloaders_cifar10(
        batch_size=16, 
        num_workers=4, 
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        validation_fraction=0.1)


    #########################################
    ### 2 Initializing the Model
    #########################################

    model = vit_b_16(weights=None)

    # replace output layer
    model.heads.head = torch.nn.Linear(in_features=768, out_features=10)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    #########################################
    ### 3 Finetuning
    #########################################

    start = time.time()
    train(
        num_epochs=10,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    end = time.time()
    elapsed = end-start
    print(f"Time elapsed {elapsed/60:.2f} min")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    #########################################
    ### 4 Evaluation
    #########################################
    
    with torch.no_grad():
        model.eval()
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

        for (features, targets) in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            predicted_labels = torch.argmax(outputs, 1)
            test_acc.update(predicted_labels, targets)

    print(f"Test accuracy {test_acc.compute()*100:.2f}%")
import time

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torchvision import transforms
from watermark import watermark

from local_utilities import get_dataloaders_cifar10


class PyTorchCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.cnn_layers = torch.nn.Sequential(

            torch.nn.Conv2d(3, 6, kernel_size=5),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            
            torch.nn.Conv2d(6, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),        
            
            torch.nn.Conv2d(16, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), 
        )
        
        self.fc_layers = torch.nn.Sequential(
            # hidden layer
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        logits = self.fc_layers(x)
        return logits


def train(num_epochs, model, optimizer, scheduler, train_loader, val_loader, device):

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
            scheduler.step()

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


    ############################################################
    #### YOUR CODE BELOW: Resize images
    ############################################################
    train_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                           transforms.ToTensor()])
    
    test_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.ToTensor()])
    
    ############################################################
    
    train_loader, val_loader, test_loader = get_dataloaders_cifar10(
        batch_size=64,
        num_workers=3,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        validation_fraction=0.1)


    #########################################
    ### 2 Initializing the Model
    #########################################


    ############################################################
    #### YOUR CODE BELOW: Replace model
    ############################################################

    # use a model from https://pytorch.org/vision/stable/models.html

    model = PyTorchCNN(num_classes=10)
    model.to(device)
    ############################################################

    NUM_EPOCHS = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    num_steps = NUM_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    #########################################
    ### 3 Finetuning
    #########################################

    start = time.time()
    train(
        num_epochs=NUM_EPOCHS,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
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
import torch
from torch import nn
from torch import optim
from torchvision import models

class Model1 (nn.Module):
    """
        The aim of this classifier is to classify Prostatic anatomopathology according to the gleason
        For each image, the output is the two most probable gleason class, the loss is a weight of :
            - Major class (0.5)
            - Minor class (0.5)
        Both are cross entropy loss.
    """
    
    def __init__ (self):
        super().__init__()
        
        # Getting the backbone
        backbone = models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[0:-1])
        
        # Getting network : deeper network
        self.network = nn.Sequential(*[
            backbone,
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        ])
        
        self.score_1 = nn.Sequential(*[
            nn.Linear(16, 6),
        ])

        self.score_2 = nn.Sequential(*[
            nn.Linear(16, 6)
        ])

        
        self.optim = optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()
    
    def forward (self, x):
        x /= 255. # Normalizing between 0 and 1
        
        z = self.network(x)
        score_1 = self.score_1(z)
        score_2 = self.score_2(z)
        
        return score_1, score_2
        
    def fit (self, x, y):
        self.train()
        
        self.optim.zero_grad()
        
        # The loss is half good image prediction, half good isup prediction
        score_1, score_2 = self.forward(x)
        loss_1 = self.criterion(score_1, y[:,0])
        loss_2 = self.criterion(score_2, y[:,1])
        loss = (loss_1 + loss_2)
        
        # The loss is weighted by the visible surface
        ## Exemple : an almost empty image should have a low loss value because it cannot be informative
        #loss_weight = (x != 1).float().mean()
        #loss = loss_weight*loss
        
        loss.backward()
        self.optim.step()
        
        loss_detached = loss.detach().cpu().item()
        
        return loss_detached
    
    def save_model (self, path):
        torch.save(self.state_dict(), path)
        
    def load_model (self, path, device="cpu"):
        return self.load_state_dict(torch.load(path, map_location=device))
        
    def predict(self, x):
        
        self.eval()
        with torch.no_grad():
            y_hat = self.forward(x)
            
            # Also returning loss_weight
            loss_weight = (x != 1).float().mean()
            
        return y_hat, loss_weight
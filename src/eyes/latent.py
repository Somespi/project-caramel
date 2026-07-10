import os
import cv2
import torch
from torch import nn, optim

class AE(nn.Module):
    def __init__(self, input_dim):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18), 
        )
        self.decoder = nn.Sequential(
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Model:
    def __init__(self, input_dim):
        self.model = AE(input_dim)
        self.input_dim = input_dim
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-8)
    
        
    def __train(self, epochs, loader):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-8)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        for epoch in range(epochs):
            last_loss = 0.0
            for images, _ in loader:
                images = images.view(-1, self.input_dim).to(device)
                predicted = self.model(images)
                loss = self.loss_function(predicted, images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {last_loss:.6f}")
            
    def generate_latent(self, frame):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval() 
        frame_norm = frame.astype("float32") / 255.0
        frame_tensor = torch.tensor(frame_norm, dtype=torch.float32).view(-1, self.input_dim).to(device)
        
        with torch.no_grad():
            latent_vector = self.model.encoder(frame_tensor)
        return latent_vector.cpu().numpy().flatten()
    
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def create_dataset_from_frames_path(self, path):
        frame_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        dataset = []
        for frame_file in frame_files:
            frame_path = os.path.join(path, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame_norm = frame.astype("float32") / 255.0
                dataset.append(frame_norm.flatten())
        return dataset
    
    

    def train_from_frames_path(self, path, epochs):
        dataset = self.create_dataset_from_frames_path(path)
        if not dataset:
            print("No valid frames found in the specified path.")
            return
        dataset_tensor = torch.tensor(dataset, dtype=torch.float32)
        from torch.utils.data import TensorDataset
        torch_dataset = TensorDataset(dataset_tensor)
        data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=32, shuffle=True)
        self.__train(data_loader, epochs)
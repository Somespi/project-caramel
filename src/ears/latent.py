import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class CNNLSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        input_length=24000,
        hidden_size=256,
        latent_size=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ---------------- CNN Encoder ---------------- #
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # ---------------- LSTM Encoder ---------------- #
        self.encoder = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Latent representation
        self.latent = nn.Linear(hidden_size, latent_size)

        # ---------------- Decoder ---------------- #
        self.hidden_proj = nn.Linear(latent_size, hidden_size)
        self.cell_proj = nn.Linear(latent_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.decoder_output = nn.Linear(hidden_size, 256)

        # CNN decoder (Transposed Convolution to upsample from 375 steps back to 24000)
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=11, stride=4, padding=5, output_padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=11, stride=4, padding=5, output_padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=11, stride=4, padding=5, output_padding=3),
            nn.Tanh()  # Tanh bounds raw audio amplitudes cleanly between -1.0 and 1.0
        )

    def encode(self, x):
        # x: (batch, 1, input_length)
        x = self.cnn_encoder(x)
        # (batch, 256, time_steps) -> (batch, time_steps, 256)
        x = x.transpose(1, 2)
        _, (h, _) = self.encoder(x)
        h = h[-1]  # Extract the final layer hidden output state
        z = self.latent(h)
        return z

    def decode(self, z):
        batch = z.size(0)

        h0 = self.hidden_proj(z)
        c0 = self.cell_proj(z)

        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)

        seq_len = 375
        decoder_input = torch.zeros(batch, seq_len, 256, device=z.device)

        decoded, _ = self.decoder(decoder_input, (h0, c0))
        decoded = self.decoder_output(decoded)

        # (batch, time_steps, 256) -> (batch, 256, time_steps)
        decoded = decoded.transpose(1, 2)
        waveform = self.cnn_decoder(decoded)
        return waveform

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
    
    
class Model:
    def __init__(
        self,
        input_length=24000, # Matches CNNLSTMAutoEncoder expectations
        hidden_size=256,
        latent_size=128,
        num_layers=2,
        dropout=0.2,
        lr=1e-3,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CNNLSTMAutoEncoder(
            input_length=input_length,
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstruction, _ = self.model(batch)

                loss = self.loss_function(reconstruction, batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
            print(f"Epoch {epoch+1}/{epochs} | Audio Loss: {avg_loss:.6f}")

    def generate_latent(self, x):
        self.model.eval()
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        # Enforce correct tensor dimension shapes: (Batch, Channels=1, Length)
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(1)

        x = x.to(self.device)
        with torch.no_grad():
            embedding = self.model.encode(x)
        return embedding.cpu().numpy().flatten()

    def save(self, path="caramel_audio.pth"):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.model.eval()

    def get_loader_from_path(self, path, batch_size=32):
        samples = []
        if not os.path.exists(path):
            return None

        for file in os.listdir(path):
            if file.endswith(".pkl"):
                with open(os.path.join(path, file), "rb") as f:
                    waveform = pickle.load(f)
                
                if not torch.is_tensor(waveform):
                    waveform = torch.tensor(waveform, dtype=torch.float32)
                
                # Make sure it's shaped explicitly as (1, 24000)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                
                samples.append(waveform)

        if len(samples) == 0:
            return None

        data = torch.stack(samples) # Shape: (N, 1, 24000)
        dataset = TensorDataset(data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_from_frames_path(self, path, epochs=10, batch_size=32):
        loader = self.get_loader_from_path(path, batch_size)
        if loader is not None:
            self.train(loader, epochs)
        else:
            print("No audio cache data (.pkl files) discovered to perform sleep loop.")
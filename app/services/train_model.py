import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from app.core.config import settings
from app.core.logger import logger


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0)


class EmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=128):
        super(EmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def create_processed_data():
    def clean_price(price):
        if isinstance(price, str):
            price = price.lower()
            if "free" in price:
                return "free"

            match = re.search(r"\$\d+(\.\d{2})?", price)
            if match:
                try:
                    return f"${float(match.group()[1:]):.2f}"
                except ValueError:
                    return np.nan
        return np.nan

    steam_data = pd.read_csv(settings.input_file_path)
    filtered_data = steam_data[
        [
            "name",
            "categories",
            "user_reviews",
            "date",
            "developer",
            "publisher",
            "price",
        ]
    ]


    filtered_data["price"] = filtered_data["price"].apply(clean_price)
    filtered_data.dropna(subset=["name", "price", "categories", "user_reviews", "date", "developer", "publisher"], inplace=True)
    sampled_data = filtered_data.copy()#.sample(n=10000, random_state=42)
    sampled_data["combined_text"] = (
        sampled_data["name"]
        + " "
        + sampled_data["categories"]
        + " "
        + sampled_data["user_reviews"].astype(str)
        + " "
        + sampled_data["date"]
        + " "
        + sampled_data["developer"]
        + " "
        + sampled_data["publisher"]
    )

    sampled_data.to_csv(settings.output_file_path, index=False)
    logger.info(f"Processed data saved to {settings.output_file_path}")
    return sampled_data


def train_autoencoder(model_name: str = "all-MiniLM-L6-v2", embeddings_size: int = 384):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    logger.info(f"Using device: {settings._device}")

    if not os.path.exists(settings.output_file_path):
        data = create_processed_data()
    else:
        data = pd.read_csv(settings.output_file_path)
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    base_model = settings.get_sentence_model(model_name)
    embeddings = base_model.encode(data['combined_text'].tolist(), convert_to_tensor=True)

    train_embeddings, val_embeddings = train_test_split(embeddings, test_size=0.2, random_state=42)
    train_data = TensorDataset(train_embeddings)
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    val_data = TensorDataset(val_embeddings)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=True)

    input_dim = embeddings.shape[1]
    autoencoder = EmbeddingAutoencoder(input_dim).to(settings._device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch[0].to(settings._device)
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


    torch.save(autoencoder.state_dict(), f"{settings.embedding_models_path}{settings.sentence_transformer_models[model_name][0]}.pth")


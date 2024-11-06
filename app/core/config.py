import os
import torch
import torch.nn as nn
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer


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


class Settings(BaseSettings):
    app_name: str = "APS2 FastAPI"
    debug_mode: bool
    db_url: str
    input_file_path: str = "data/raw/steam_data.csv"
    output_file_path: str = "data/processed/steam_data.csv"
    embeddings_size: int
    embedding_models_path: str
    sentence_transformer_models: dict = {
        "MiniLM-L6-v2": ("all-MiniLM-L6-v2", 384),
        "MiniLM-L12-v2": ("all-MiniLM-L12-v2", 384),
        "MPNet-base-v2": ("all-mpnet-base-v2", 768),
        "DistilBERT-base-STSB": ("distilbert-base-nli-stsb-mean-tokens", 768),
        "RoBERTa-distil-base": ("paraphrase-distilroberta-base-v1", 768),
        "RoBERTa-multilingual": ("paraphrase-xlm-r-multilingual-v1", 768),
        "BERT-base-NLI": ("bert-base-nli-mean-tokens", 768),
        "BERT-large-NLI": ("bert-large-nli-mean-tokens", 1024),
        "MiniLM-multilingual-L12": ("paraphrase-multilingual-MiniLM-L12-v2", 384),
        "DistilUSE-multilingual-cased": ("distiluse-base-multilingual-cased-v2", 512),
        "MiniLM-L3-v2": ("paraphrase-MiniLM-L3-v2", 384)
    }

    _sentence_model: str = None
    _autoencoder: str = None
    _device: str = None


    def __init__(self):
        super().__init__()
        self._device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    def get_sentence_model(self, model_name="all-MiniLM-L6-v2"):
        if self._sentence_model is None:
            self._sentence_model = SentenceTransformer(model_name).to(self._device)
        return self._sentence_model

    def get_autoencoder(self, model_name="all-MiniLM-L6-v2", embeddings_size=384):
        if self._autoencoder is None:
            self._autoencoder = EmbeddingAutoencoder(input_dim=embeddings_size).to(self._device)
            if not os.path.exists(f"{self.embedding_models_path}{model_name}.pth"):
                raise FileNotFoundError(f"Model file {self.embedding_models_path}{model_name}.pth not found.")
            self._autoencoder.load_state_dict(torch.load(f"{self.embedding_models_path}{model_name}.pth", map_location=self._device))
            self._autoencoder.eval()
        return self._autoencoder

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

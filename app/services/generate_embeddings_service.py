from typing import Tuple
import torch
import torch.nn as nn
from app.core.config import settings


class EmbeddingService:
    def __init__(self, model_setup: Tuple[str, str]):
        self.sentence_model = settings.get_sentence_model(model_setup[0])
        self.autoencoder = settings.get_autoencoder(model_setup[0], model_setup[1])

    def encode_texts(self, texts: list):
        embeddings = self.sentence_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.to(settings._device)
        return embeddings

    def fine_tune_embeddings(self, embeddings):
        with torch.no_grad():
            fine_tuned_embeddings = self.autoencoder(embeddings).cpu()
        return fine_tuned_embeddings

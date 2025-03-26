import torch
from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:
    """
    Class responsible for generating embeddings from text.

    Args:
        model_name (str): Name of the model to use for embeddings.

    Attributes:
        model_name (str): Name of the model to use for embeddings.
        embedding (HuggingFaceEmbeddings): Loaded Hugging Face embedding model.
    """

    def __init__(
        self,
        model_name: str = "bert-large-uncased"
    ):
        self.model_name = model_name
        self.embedding = self._load_embedding_model()

    def _load_embedding_model(self) -> HuggingFaceEmbeddings:
        """
        Loads the embedding model based on the model name.

        Returns:
            HuggingFaceEmbeddings: Loaded Hugging Face embedding model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(
            model_name=self.model_name, model_kwargs={"device": device}
        )

    def generate_embeddings(self, text: str) -> list:
        """
        Generate embeddings for a given text.

        Args:
            text (str): Text to generate embeddings for.

        Returns:
            list: List of embedding vectors.
        """
        return self.embedding.embed_documents([text])



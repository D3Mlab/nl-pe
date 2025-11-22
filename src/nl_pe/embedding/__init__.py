from nl_pe.embedding.embedders import HuggingFaceEmbedderSentenceTransformers
from nl_pe.embedding.embedders import GoogleEmbedder

EMBEDDER_CLASSES = {
    "HuggingFaceEmbedderSentenceTransformers": HuggingFaceEmbedderSentenceTransformers,
    "GoogleEmbedder": GoogleEmbedder
}

from nl_pe.embedding.embedders import HuggingFaceEmbedderSentenceTransformers
from nl_pe.embedding.embedders import GoogleEmbedder
from nl_pe.embedding.embedders import DimTruncator


EMBEDDER_CLASSES = {
    "HuggingFaceEmbedderSentenceTransformers": HuggingFaceEmbedderSentenceTransformers,
    "GoogleEmbedder": GoogleEmbedder,
    "DimTruncator": DimTruncator
}

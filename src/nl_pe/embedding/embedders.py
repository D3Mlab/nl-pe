"""
Embedding module with GPU-accelerated KNN operations and comprehensive debug logging.

KNN Methods:
- exact_knn_from_embeddings: Full GPU acceleration for torch.save embeddings
- exact_knn_from_faiss: Optimized FAISS search (no CSV reading)
- exact_knn_from_db: Memory-efficient batched processing for shelve databases

All methods assume document IDs are 0 to N-1 in CSV order for optimal performance.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import shelve
import numpy as np


class BaseEmbedder(ABC):

    def __init__(self, normalize=True):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.normalize = normalize

    @abstractmethod
    def embed_documents_batch(self, texts: list[str], prompt = '') -> Tensor:
        """
        Embed a batch of texts into torch tensor, normalized if self.normalize.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    def save_embeddings_torch(self, texts_csv_path = '', index_path = '', batch_size = None, prompt = ''):
        """
        Reads a CSV where the first column is the text to embed.
        Saves only the embeddings tensor to index_path for maximum efficiency.
        """
        self.logger.debug(f"Reading texts from CSV: {texts_csv_path}")
        df = pd.read_csv(texts_csv_path, header=0)
        texts = df.iloc[:, 0].tolist()
        self.logger.debug(f"First 3 texts: {texts[:3]}... ")

        self.logger.debug(f"Loading {len(texts)} documents from CSV for embedding")

        num_docs = len(texts)
        batch_size = batch_size or num_docs
        all_embeddings = []

        for i in range(0, num_docs, batch_size):
            batch_texts = texts[i:i + batch_size]
            self.logger.debug(f"Processing embedding batch {i//batch_size + 1}/{(num_docs + batch_size - 1) // batch_size} with {len(batch_texts)} documents")

            embeddings_tensor = self.embed_documents_batch(batch_texts)
            self.logger.debug(f"Embeddings tensor device: {embeddings_tensor.device}, shape: {embeddings_tensor.shape}")

            all_embeddings.append(embeddings_tensor)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        self.logger.debug(f"Final embeddings shape: {all_embeddings.shape}, device: {all_embeddings.device}")
        
        # Save tensor directly from GPU
        self.logger.debug(f"Saving embeddings tensor (device: {all_embeddings.device}) to {index_path}")
        torch.save(all_embeddings, index_path)
        self.logger.info("Saved embeddings tensor to %s", index_path)

    def embed_all_docs_faiss_exact(self, texts_csv_path='', index_path='', batch_size=None, prompt = ''):
        """Embed all documents and create an exact FAISS index."""
        self.logger.debug(f"Creating FAISS index from {texts_csv_path}")
        df = pd.read_csv(texts_csv_path, header=0)
        texts = df.iloc[:, 0].tolist()

        num_docs = len(texts)
        batch_size = batch_size or num_docs
        self.logger.debug(f"Processing {num_docs} docs in batches of {batch_size}")

        # Embed first batch to get dimensionality and add to index
        first_batch = self.embed_documents_batch(texts[:batch_size])
        d = first_batch.shape[1]
        self.logger.debug(f"Embedding dimension: {d}")

        index = faiss.IndexFlatIP(d)  # Inner product
        use_gpu = faiss.get_num_gpus() > 0
        if use_gpu:
            self.logger.debug("Using FAISS GPU acceleration")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        #add first batch
        batch_embeddings = first_batch.detach().cpu().numpy()
        index.add(batch_embeddings)
        self.logger.debug("Added first batch to FAISS index")

        # Process remaining batches
        for i in range(batch_size, num_docs, batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings_tensor = self.embed_documents_batch(batch_texts)
            batch_embeddings = embeddings_tensor.detach().cpu().numpy()
            index.add(batch_embeddings)
            self.logger.debug(f"Added batch {i//batch_size + 1}/{(num_docs + batch_size - 1)//batch_size}")

        # Move index back to CPU if GPU was used and save
        if use_gpu:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, index_path)
        self.logger.info(f"Saved FAISS exact index to {index_path}")



    def embed_doc_batches_db(self, texts_csv_path='', index_path='', batch_size=10000, prompt = ''):
        self.logger.debug(f"Creating shelve database from {texts_csv_path}")

        with shelve.open(index_path, writeback=False) as db:
            df = pd.read_csv(texts_csv_path, header=0)
            num_docs = len(df)
            self.logger.debug(f"Loaded {num_docs} documents for shelve storage")

            if not batch_size:
                batch_size = num_docs

            for i in range(0, num_docs, batch_size):
                batch_df = df.iloc[i:i + batch_size]
                texts = batch_df.iloc[:, 0].tolist()

                self.logger.debug(f"Processing shelve batch {i//batch_size + 1}/{(num_docs + batch_size - 1) // batch_size} with {len(texts)} documents")

                embeddings_tensor = self.embed_documents_batch(texts)
                self.logger.debug(f"Batch embeddings device: {embeddings_tensor.device}, shape: {embeddings_tensor.shape}")

                # Keep embeddings on GPU and use torch.save for each embedding
                for j, emb in enumerate(embeddings_tensor):
                    doc_id = i + j # Implicit doc_id is the row index
                    # Use torch.save to preserve GPU tensor without CPU conversion
                    import io
                    buffer = io.BytesIO()
                    torch.save(emb, buffer)
                    buffer.seek(0)
                    db[str(doc_id)] = buffer.read()  # Store raw bytes
                    self.logger.debug(f"Stored GPU embedding for doc_id {doc_id} using torch.save")

                db.sync()  # force flush to disk
                self.logger.info("Processed batch %d/%d",
                                (i // batch_size) + 1,
                                (num_docs + batch_size - 1) // batch_size)

            self.logger.info("Saved embeddings to shelve db %s", index_path)

    def exact_knn_from_torch_all_in_mem(self, query: str, k=10, prompt='', embeddings_path='') -> list[int]:
        #for small corpora only, loads all embeddings to GPU/CPU and uses a single matrix-vector multiply

        self.logger.debug(f"Starting KNN search for query with k={k}")

        # Embed the query
        query_emb = self.embed_documents_batch([query], prompt=prompt)[0]
        self.logger.debug(f"query_emb device after embedding: {query_emb.device}, shape={query_emb.shape}")

        # Load embeddings directly as a single tensor onto GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Loading embeddings tensor from {embeddings_path} to {device}")
        embeddings_tensor = torch.load(embeddings_path, map_location=device)
        self.logger.debug(f"Loaded embeddings tensor: shape={embeddings_tensor.shape}, device={embeddings_tensor.device}")

        # Move query to same device
        query_emb = query_emb.to(device)

        # Compute similarities
        similarities = torch.mv(embeddings_tensor, query_emb)
        self.logger.debug(f"similarities: device={similarities.device}, shape={similarities.shape}")

        # Top-k retrieval
        _, top_k_indices = torch.topk(similarities, k, largest=True)
        self.logger.debug(f"Top-k indices: {top_k_indices.tolist()}")

        return top_k_indices.tolist()  

    def exact_knn_from_faiss(self, query: str, k=10, prompt='', index_path='') -> list[str]:
        query_emb = self.embed_documents_batch([query], prompt=prompt)[0]
        self.logger.debug(f"query_emb device after embedding: {query_emb.device}")

        # Since doc IDs are 0 to N-1 in CSV order, we can generate them directly from indices
        index = faiss.read_index(index_path)
        query_emb_np = query_emb.detach().cpu().numpy().reshape(1, -1)
        self.logger.debug(f"query_emb_np created from query_emb.cpu().numpy(), shape: {query_emb_np.shape}")

        distances, indices = index.search(query_emb_np, k)
        self.logger.debug(f"FAISS search completed, found {len(indices[0])} results")

        # Convert indices directly to doc IDs (0 to N-1)
        result_doc_ids = [str(i) for i in indices[0]]
        self.logger.debug(f"Final FAISS results: {result_doc_ids}")

        return result_doc_ids

    def exact_knn_from_db(self, query: str, k=10, prompt='', db_path='', batch_size=1000) -> list[str]:
        """
        Memory-efficient KNN retrieval using batched processing.
        Only keeps top-k (or batch_size) highest scoring documents in memory.
        """
        query_emb = self.embed_documents_batch([query], prompt=prompt)[0]
        self.logger.debug(f"query_emb device after embedding: {query_emb.device}")

        query_np = query_emb.detach().cpu().numpy()
        self.logger.debug(f"query_np created from query_emb.cpu().numpy()")

        # Use a max heap to track top-k results efficiently
        import heapq
        top_k_heap = []  # Will store (negative similarity, doc_id) for min-heap behavior

        with shelve.open(db_path) as db:
            # Process in batches to avoid loading all embeddings at once
            all_keys = list(db.keys())
            total_docs = len(all_keys)
            self.logger.debug(f"Processing {total_docs} total documents in batches of {batch_size}")

            for i in range(0, total_docs, batch_size):
                batch_keys = all_keys[i:i + batch_size]
                batch_embeddings = []
                self.logger.debug(f"Processing batch {i//batch_size + 1}, keys: {batch_keys[:3]}...")  # Show first 3 keys

                # Load batch embeddings using torch.load to preserve GPU tensors
                for key in batch_keys:
                    emb_bytes = db[key]
                    import io
                    buffer = io.BytesIO(emb_bytes)
                    emb = torch.load(buffer)
                    self.logger.debug(f"Loaded embedding for doc_id {key}, device: {emb.device}")
                    batch_embeddings.append(emb.detach().cpu().numpy())

                # Vectorized similarity computation for this batch with PyTorch GPU acceleration
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.logger.debug(f"Batch processing - Target device: {device}")

                batch_embeddings_tensor = torch.tensor(batch_embeddings, dtype=torch.float32).to(device)
                self.logger.debug(f"batch_embeddings_tensor device after .to(device): {batch_embeddings_tensor.device}")

                query_tensor = torch.tensor(query_np, dtype=torch.float32).to(device)
                self.logger.debug(f"query_tensor device after .to(device): {query_tensor.device}")

                batch_similarities = torch.mv(batch_embeddings_tensor, query_tensor).tolist()
                self.logger.debug(f"batch_similarities computed, length: {len(batch_similarities)}")

                # Update top-k results for this batch
                for key, similarity in zip(batch_keys, batch_similarities):
                    heapq.heappush(top_k_heap, (similarity, key))
                    # Keep only top-k (or batch_size) items in heap
                    if len(top_k_heap) > max(k, batch_size):
                        heapq.heappop(top_k_heap)

        # Extract final top-k results (heap contains lowest scores first)
        # We need to sort by similarity (descending) to get actual top-k
        top_k_results = []
        while top_k_heap:
            similarity, doc_id = heapq.heappop(top_k_heap)
            top_k_results.append((similarity, doc_id))

        # Sort by similarity (highest first) and return top-k doc IDs
        top_k_results.sort(reverse=True, key=lambda x: x[0])
        self.logger.debug(f"Final top-k results: {[doc_id for _, doc_id in top_k_results[:k]]}")

        return [doc_id for _, doc_id in top_k_results[:k]]

class HuggingFaceEmbedderSentenceTransformers(BaseEmbedder):

    def __init__(self, model_name='', matryoshka_dim=None, normalize=True):
        super().__init__(normalize)

        # e.g. model_name = Qwen/Qwen3-Embedding-8B
        self.model = SentenceTransformer(
            model_name,
            #no flash attention, since may need wsl switch
            model_kwargs={}, #{"device_map": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
            truncate_dim=matryoshka_dim,
        )

        device = getattr(self.model, "device", getattr(self.model, "device", "unknown"))
        self.logger.info(f"Model device: {device}")
        self.logger.info("Matryoshka dimension set to: %s", matryoshka_dim if matryoshka_dim else "full")

    def embed_documents_batch(self, texts: list[str], prompt = '') -> Tensor:
        self.logger.debug(f"Encoding {len(texts)} texts in batch")

        kwargs = dict(
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True
        )

        if prompt:
            self.logger.debug(f"Using prompt: {prompt}")
            self.model.prompts['prompt'] = prompt
            kwargs["prompt_name"] = "prompt"

        embeddings_tensor = self.model.encode(texts, **kwargs)

        self.logger.debug(f"Embeddings created, device: {embeddings_tensor.device}, shape: {embeddings_tensor.shape}")
        return embeddings_tensor


# For transformers package use
    @staticmethod
    def _last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

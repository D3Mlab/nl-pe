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
import os
import heapq, io, torch, shelve
import time
from nl_pe.utils.setup_logging import setup_logging

class BaseEmbedder(ABC):

    def __init__(self, config):
        self.config = config
        self.embedding_config = self.config.get('embedding', {})
        self.data_config = self.config.get('data', {})
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))
        self.normalize = self.embedding_config.get('normalize', True)
        self.model_name = self.embedding_config.get('model', '')
        self.matryoshka_dim = self.embedding_config.get('matryoshka_dim', None)
        self.embeddings_path = self.data_config.get('index_path', '')

        #knn:
        self.k = self.embedding_config.get('k')
        self.similarity_batch_size = self.embedding_config.get('similarity_batch_size')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def embed_documents_batch(self, texts: list[str], prompt = '') -> Tensor:
        """
        Embed a batch of texts into torch tensor, normalized if self.normalize.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    def save_embeddings_torch(self, texts_csv_path = '', index_path = '', inference_batch_size = None, prompt = ''):
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
        inference_batch_size = inference_batch_size or num_docs
        all_embeddings = []

        for i in range(0, num_docs, inference_batch_size):
            batch_texts = texts[i:i + inference_batch_size]
            self.logger.debug(f"Processing embedding batch {i//inference_batch_size + 1}/{(num_docs + inference_batch_size - 1) // inference_batch_size} with {len(batch_texts)} documents")

            embeddings_tensor = self.embed_documents_batch(batch_texts, prompt = prompt)
            self.logger.debug(f"Embeddings tensor device: {embeddings_tensor.device}, shape: {embeddings_tensor.shape}")

            all_embeddings.append(embeddings_tensor)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        self.logger.debug(f"Final embeddings shape: {all_embeddings.shape}, device: {all_embeddings.device}")
        
        # Save tensor directly from GPU
        self.logger.debug(f"Saving embeddings tensor (device: {all_embeddings.device}) to {index_path}")
        torch.save(all_embeddings, index_path)
        self.logger.info("Saved embeddings tensor to %s", index_path)

    def embed_all_docs_faiss_exact(self, texts_csv_path='', index_path='', inference_batch_size=None, prompt = ''):
        """Embed all documents and create an exact FAISS index."""
        self.logger.debug(f"Creating FAISS index from {texts_csv_path}")
        df = pd.read_csv(texts_csv_path, header=0)
        texts = df.iloc[:, 0].tolist()

        num_docs = len(texts)
        inference_batch_size = inference_batch_size or num_docs
        self.logger.debug(f"Processing {num_docs} docs in batches of {inference_batch_size}")

        # Embed first batch to get dimensionality and add to index
        first_batch = self.embed_documents_batch(texts[:inference_batch_size], prompt=prompt)
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
        for i in range(inference_batch_size, num_docs, inference_batch_size):
            batch_texts = texts[i:i + inference_batch_size]
            embeddings_tensor = self.embed_documents_batch(batch_texts, prompt=prompt)
            batch_embeddings = embeddings_tensor.detach().cpu().numpy()
            index.add(batch_embeddings)
            self.logger.debug(f"Added batch {i//inference_batch_size + 1}/{(num_docs + inference_batch_size - 1)//inference_batch_size}")

        # Move index back to CPU if GPU was used and save
        if use_gpu:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, index_path)
        self.logger.info(f"Saved FAISS exact index to {index_path}")

    def embed_doc_batches_db(self, texts_csv_path='', index_path='', inference_batch_size=None, prompt = ''):
        self.logger.debug(f"Creating shelve database from {texts_csv_path}")

        with shelve.open(index_path, writeback=False) as db:
            df = pd.read_csv(texts_csv_path, header=0)
            num_docs = len(df)
            self.logger.debug(f"Loaded {num_docs} documents for shelve storage")

            if not inference_batch_size:
                inference_batch_size = num_docs

            for i in range(0, num_docs, inference_batch_size):
                batch_df = df.iloc[i:i + inference_batch_size]
                texts = batch_df.iloc[:, 0].tolist()

                self.logger.debug(f"Processing shelve batch {i//inference_batch_size + 1}/{(num_docs + inference_batch_size - 1) // inference_batch_size} with {len(texts)} documents")

                embeddings_tensor = self.embed_documents_batch(texts, prompt=prompt)
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
                                (i // inference_batch_size) + 1,
                                (num_docs + inference_batch_size - 1) // inference_batch_size)

            self.logger.info("Saved embeddings to shelve db %s", index_path)

    def exact_knn_from_torch_all_in_mem(self, state) -> list[int]:
        #for small corpora only, loads all embeddings to GPU/CPU and uses a single matrix-vector multiply
        start_time = time.time()
        self.logger.debug(f"Starting KNN search for query with k={self.k}")

        query = state.get("query")    

        # Embed the query
        query_emb = self.embed_documents_batch([query], prompt=self.embedding_config.get("query_prompt", ''))[0]
        self.logger.debug(f"query_emb device after embedding: {query_emb.device}, shape={query_emb.shape}")

        # Load embeddings directly as a single tensor onto GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Loading embeddings tensor from {self.embeddings_path} to {device}")
        embeddings_tensor = torch.load(self.embeddings_path, map_location=device)
        self.logger.debug(f"Loaded embeddings tensor: shape={embeddings_tensor.shape}, device={embeddings_tensor.device}")

        # Move query to same device
        query_emb = query_emb.to(device)

        # Compute similarities
        similarities = torch.mv(embeddings_tensor, query_emb)
        self.logger.debug(f"similarities: device={similarities.device}, shape={similarities.shape}")

        # Top-k retrieval
        top_k_scores, top_k_indices = torch.topk(similarities, self.k, largest=True)
        self.logger.debug(f"Top-k indices: {top_k_indices.tolist()}")

        state['top_k_psgs'] = top_k_indices.tolist()
        state['knn_scores'] = top_k_scores.tolist()
        state['knn_time'] = time.time() - start_time

        #del embeddings_tensor, query_emb, similarities
        #torch.cuda.empty_cache()

    def exact_knn_from_faiss(self, state) -> list[str]:
        start_time = time.time()
        query = state.get("query")

        query_emb = self.embed_documents_batch([query], prompt=self.embedding_config.get("query_prompt", ''))[0]
        self.logger.debug(f"query_emb device after embedding: {query_emb.device}")

        # Since doc IDs are 0 to N-1 in CSV order, we can generate them directly from indices
        index = faiss.read_index(self.embeddings_path)
        query_emb_np = query_emb.detach().cpu().numpy().reshape(1, -1)
        self.logger.debug(f"query_emb_np created from query_emb.cpu().numpy(), shape: {query_emb_np.shape}")

        distances, indices = index.search(query_emb_np, self.k)
        self.logger.debug(f"FAISS search completed, found {len(indices[0])} results")
        state['top_k_psgs'] = indices[0].tolist()
        state['knn_scores'] = distances[0].tolist()
        state['knn_time'] = time.time() - start_time

    def exact_knn_from_db(self, state) -> list[str]:
        """
        Batched KNN retrieval:
        - Processing embeddings in GPU batches
        - Keeping only top-k results in a bounded heap
        """
        start_time = time.time()

        query = state.get("query")
        query_emb = self.embed_documents_batch([query], prompt=self.embedding_config.get("query_prompt", ''))[0]
        query_np = query_emb.detach().cpu().numpy()
        k = getattr(self, "k", 10)  # fallback default if not set

        top_k_heap = []  # min-heap of (similarity, doc_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with shelve.open(self.embeddings_path, "r") as db:
            all_keys = list(db.keys())
            total_docs = len(all_keys)
            self.logger.debug(f"Processing {total_docs} total docs in batches of {self.similarity_batch_size}")

            for i in range(0, total_docs, self.similarity_batch_size):
                batch_keys = all_keys[i:i + self.similarity_batch_size]

                # --- Load and stack batch embeddings ---
                batch_embs = []
                for key in batch_keys:
                    buffer = io.BytesIO(db[key])
                    emb = torch.load(buffer, map_location="cpu").float()
                    batch_embs.append(emb)

                batch_tensor = torch.stack(batch_embs).to(device)
                query_tensor = torch.tensor(query_np, dtype=torch.float32, device=device)

                # --- Compute cosine similarity (vectorized) ---
                sim = torch.mv(batch_tensor, query_tensor).tolist()

                # --- Update top-k heap ---
                for key, s in zip(batch_keys, sim):
                    if len(top_k_heap) < k:
                        heapq.heappush(top_k_heap, (s, key))
                    else:
                        heapq.heappushpop(top_k_heap, (s, key))

                del batch_embs, batch_tensor  # free GPU memory
                torch.cuda.empty_cache()

                self.logger.debug(f"Processed batch {i//self.similarity_batch_size + 1}/{(total_docs - 1)//self.similarity_batch_size + 1}")

        # --- Sort heap descending by similarity ---
        top_k_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
        self.logger.debug(f"Final top-{k} doc_ids: {[doc_id for _, doc_id in top_k_results]}")
        state['top_k_psgs'] = [doc_id for _, doc_id in top_k_results]
        state['knn_scores'] = [score for score, _ in top_k_results]
        state['knn_time'] = time.time() - start_time



class HuggingFaceEmbedderSentenceTransformers(BaseEmbedder):

    def __init__(self, config):
        super().__init__(config)

        # e.g. model_name = Qwen/Qwen3-Embedding-8B
        self.model = SentenceTransformer(
            self.model_name,
            #no flash attention, since may need wsl switch
            model_kwargs={}, #{"device_map": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
            truncate_dim=self.matryoshka_dim,
            device = self.device
        )

        device = getattr(self.model, "device", getattr(self.model, "device", "unknown"))
        self.logger.info(f"Model device: {device}")
        self.logger.info("Matryoshka dimension set to: %s", self.matryoshka_dim if self.matryoshka_dim else "full")

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

        embeddings_tensor = self.model.encode(texts, **kwargs) #.to(device)

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
            inference_batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(inference_batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

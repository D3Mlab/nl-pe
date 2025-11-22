"""
Embedding module with GPU-accelerated KNN operations and comprehensive debug logging.

KNN Methods:
- exact_knn_from_embeddings: Full GPU acceleration for torch.save embeddings with arbitrary document IDs from CSV first column
- exact_knn_from_faiss: Optimized FAISS search with arbitrary document IDs from CSV first column
- exact_knn_from_db: Memory-efficient batched processing for shelve databases with arbitrary document IDs
"""

#from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import shelve
import pickle
import numpy as np
import os
import heapq, io, torch, shelve
import time
from nl_pe.utils.setup_logging import setup_logging
import gc
import copy
from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPIError

class BaseEmbedder(ABC):

    def __init__(self, config):
        #free_gpu_memory()

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

        self.inference_device = normalize_device(self.config.get('inference_device', self.config.get('device', 'cpu')))
        self.tensor_ops_device = normalize_device(self.config.get('tensor_ops_device', self.config.get('device', 'cpu')))

    @abstractmethod
    def embed_documents_batch(self, texts: list[str]) -> Tensor:
        """
        Embed a batch of texts into torch tensor, normalized if self.normalize.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    def save_embeddings_torch(self, texts_csv_path = '', index_path = '', inference_batch_size = None, prompt = ''):
        """
        Reads a CSV where the first column is the doc_id and second the text to embed.
        Saves the embeddings tensor and doc_ids pickle to index_path for maximum efficiency.
        """
        self.logger.debug(f"Reading texts from CSV: {texts_csv_path}")
        df = pd.read_csv(texts_csv_path, header=0)
        doc_ids = df.iloc[:, 0].tolist()
        texts = df.iloc[:, 1].tolist()
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
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        torch.save(all_embeddings, index_path)
        pickle.dump(doc_ids, open(index_path + "_doc_ids.pkl", 'wb'))
        self.logger.info("Saved embeddings tensor and doc IDs to %s", index_path)

    def embed_all_docs_faiss_exact(self, texts_csv_path='', index_path='', inference_batch_size=None, prompt = ''):
        """Embed all documents and create an exact FAISS index."""
        self.logger.debug(f"Creating FAISS index from {texts_csv_path}")
        df = pd.read_csv(texts_csv_path, header=0)
        doc_ids = df.iloc[:, 0].tolist()
        texts = df.iloc[:, 1].tolist()

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
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        pickle.dump(doc_ids, open(index_path + "_doc_ids.pkl", 'wb'))
        self.logger.info(f"Saved FAISS exact index and doc IDs to {index_path}")

    def embed_doc_batches_db(self, texts_csv_path='', index_path='', inference_batch_size=None, prompt = ''):
        self.logger.debug(f"Creating shelve database from {texts_csv_path}")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        with shelve.open(index_path, writeback=False) as db:
            df = pd.read_csv(texts_csv_path, header=0)
            num_docs = len(df)
            self.logger.debug(f"Loaded {num_docs} documents for shelve storage")

            if not inference_batch_size:
                inference_batch_size = num_docs

            for i in range(0, num_docs, inference_batch_size):
                batch_df = df.iloc[i:i + inference_batch_size]
                doc_ids_batch = batch_df.iloc[:, 0].tolist()
                texts = batch_df.iloc[:, 1].tolist()

                self.logger.debug(f"Processing shelve batch {i//inference_batch_size + 1}/{(num_docs + inference_batch_size - 1) // inference_batch_size} with {len(texts)} documents")

                embeddings_tensor = self.embed_documents_batch(texts, prompt=prompt)
                self.logger.debug(f"Batch embeddings device: {embeddings_tensor.device}, shape: {embeddings_tensor.shape}")

                # Keep embeddings on GPU and use torch.save for each embedding
                for j, emb in enumerate(embeddings_tensor):
                    doc_id = doc_ids_batch[j]  # Use the actual doc_id from CSV
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

    def exact_knn_from_torch_all_in_mem(self, state) -> list[str]:
        #for small corpora only, loads all embeddings to GPU/CPU and uses a single matrix-vector multiply
        start_time = time.time()
        self.logger.debug(f"Starting KNN search for query with k={self.k}")

        query = state.get("query")

        # Embed the query
        query_emb = self.embed_documents_batch([query], prompt=self.embedding_config.get("query_prompt", ''))[0]
        self.logger.debug(f"query_emb device after embedding: {query_emb.device}, shape={query_emb.shape}")

        # Load embeddings directly as a single tensor onto tensor_ops_device
        device = torch.device(self.tensor_ops_device)
        self.logger.debug(f"Loading embeddings tensor from {self.embeddings_path} to {device}")
        embeddings_tensor = torch.load(self.embeddings_path, map_location=device)
        self.logger.debug(f"Loaded embeddings tensor: shape={embeddings_tensor.shape}, device={embeddings_tensor.device}")

        # Load doc_ids from pickle file
        doc_ids = pickle.load(open(self.embeddings_path + "_doc_ids.pkl", 'rb'))

        # Move query to tensor_ops_device
        query_emb = query_emb.to(device)

        # Compute similarities
        similarities = torch.mv(embeddings_tensor, query_emb)
        self.logger.debug(f"similarities: device={similarities.device}, shape={similarities.shape}")

        # Top-k retrieval
        top_k_scores, top_k_indices = torch.topk(similarities, self.k, largest=True)
        self.logger.debug(f"Top-k indices: {top_k_indices.tolist()}")

        state['top_k_psgs'] = [doc_ids[i] for i in top_k_indices.tolist()]
        state['init_knn_pid_list'] = copy.deepcopy(state['top_k_psgs'])
        state['knn_scores'] = top_k_scores.tolist()
        state['knn_time'] = time.time() - start_time

        #del embeddings_tensor, query_emb, similarities
        #torch.cuda.empty_cache()

    def exact_knn_from_faiss(self, state) -> list[str]:
        start_time = time.time()
        query = state.get("query")

        query_emb = self.embed_documents_batch([query], prompt=self.embedding_config.get("query_prompt", ''))[0]
        self.logger.debug(f"query_emb device after embedding: {query_emb.device}")

        # Load doc_ids from pickle file
        doc_ids = pickle.load(open(self.embeddings_path + "_doc_ids.pkl", 'rb'))
        index = faiss.read_index(self.embeddings_path)
        query_emb_np = query_emb.detach().cpu().numpy().reshape(1, -1)
        self.logger.debug(f"query_emb_np created from query_emb.cpu().numpy(), shape: {query_emb_np.shape}")

        distances, indices = index.search(query_emb_np, self.k)
        self.logger.debug(f"FAISS search completed, found {len(indices[0])} results")
        state['top_k_psgs'] = [doc_ids[i] for i in indices[0]]
        state['init_knn_pid_list'] = copy.deepcopy(state['top_k_psgs'])
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
        device = torch.device(self.tensor_ops_device)

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
        state['init_knn_pid_list'] = copy.deepcopy(state['top_k_psgs'])
        state['knn_scores'] = [score for score, _ in top_k_results]
        state['knn_time'] = time.time() - start_time

#note google embeddings can use a document and query argument during embedding, to do later
class GoogleEmbedder(BaseEmbedder):
    
    def __init__(self, config):
        super().__init__(config)
        self.logger.info(f"Model device: {self.inference_device}")
        self.logger.info("Matryoshka dimension set to: %s", self.matryoshka_dim if self.matryoshka_dim else "full")

        self.client = genai.Client()

    def embed_documents_batch(self, texts: list[str], prompt = '') -> Tensor:
        #add 'task types' later
        self.logger.debug(f"Encoding {len(texts)} texts in batch")


        try:
            result = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
            )
        except Exception as e:
            self.logger.warning(
                f"Google embed_content error: {e}. Sleeping 65s then retrying once."
            )
            time.sleep(65)
            # single retry
            try:
                result = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                )
            except Exception as e:
                self.logger.warning(
                    f"Google embed_content error on retry: {e}. Giving up."
                )
                raise

        embedding_values = [emb.values for emb in result.embeddings]

        embeddings_tensor = torch.tensor(
            embedding_values,
            dtype=torch.float32,
            device=self.inference_device,
        )

        self.logger.debug(
            "Google embeddings created, device: %s, shape: %s",
            embeddings_tensor.device,
            tuple(embeddings_tensor.shape),
        )

        return embeddings_tensor


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
            device = self.inference_device,
            trust_remote_code=True,  # <-- add this
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


def free_gpu_memory():
    # Delete old models / variables
    gc.collect()
    torch.cuda.empty_cache()
    # Optional: force garbage collection for Python objects
    torch.cuda.synchronize()


# Device configuration: embedding operations use inference_device, tensor ops use tensor_ops_device
def normalize_device(dev):
    if dev == 'gpu' or dev == 'cuda':
        return 'cuda:0'
    elif dev == 'cpu':
        return 'cpu'
    else:
        # Assume it's already a proper device string like 'cuda:1'
        return dev

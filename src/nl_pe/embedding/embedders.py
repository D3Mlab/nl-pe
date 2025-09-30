from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
import logging
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import shelve
import numpy as np

class BaseEmbedder(ABC):

    def __init__(self, normalize_embeddings=True):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.normalize = normalize_embeddings

    @abstractmethod
    def embed_documents_batch(self, texts: list[str]) -> Tensor:
        """
        Embed a batch of texts into torch tensor, normalized if self.normalize.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    def embed_all_docs_pckl(self, texts_csv_path = '', index_path = '', batch_size = None):
        #inputs: path to csv with headers where first col is doc id and second is doc text
        #output, pickled dict of {<d_id>: <torch_embedding>}
        df = pd.read_csv(texts_csv_path, header=0)
        doc_ids = df.iloc[:, 0].tolist()
        texts = df.iloc[:, 1].tolist()

        embedding_dict = {}
        num_docs = len(doc_ids)
        batch_size = batch_size or num_docs  # if none, process all
        for i in range(0, num_docs, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = doc_ids[i:i + batch_size]
            embeddings_tensor = self.embed_documents_batch(batch_texts)
            embeddings = list(embeddings_tensor)
            for emb, doc_id in zip(embeddings, batch_ids):
                embedding_dict[doc_id] = emb

        with open(index_path, "wb") as f:
            pickle.dump(embedding_dict, f)
        self.logger.info("Saved embeddings to %s", index_path)

    def embed_all_docs_faiss_exact(self, texts_csv_path = '', index_path = '', batch_size = None):
        df = pd.read_csv(texts_csv_path, header=0)
        doc_ids = df.iloc[:, 0].tolist()
        texts = df.iloc[:, 1].tolist()

        num_docs = len(texts)
        batch_size = batch_size or num_docs
        all_embeddings = []
        for i in range(0, num_docs, batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings_tensor = self.embed_documents_batch(batch_texts)
            all_embeddings.append(embeddings_tensor.detach().cpu().numpy())
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        d = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            index.add(all_embeddings)
            cpu_index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(cpu_index, index_path)
        else:
            index.add(all_embeddings)
            faiss.write_index(index, index_path)
        self.logger.info("Saved FAISS exact index to %s", index_path)

    def embed_doc_batches_db(self, texts_csv_path='', index_path='', batch_size=10000):
        with shelve.open(index_path, writeback=False) as db:
            df = pd.read_csv(texts_csv_path, header=0)
            num_docs = len(df)

            for i in range(0, num_docs, batch_size):
                batch_df = df.iloc[i:i + batch_size]
                doc_ids = batch_df.iloc[:, 0].tolist()
                texts = batch_df.iloc[:, 1].tolist()

                embeddings_tensor = self.embed_documents_batch(texts)
                embeddings = embeddings_tensor.detach().cpu()

                for doc_id, emb in zip(doc_ids, embeddings):
                    db[str(doc_id)] = emb

                db.sync()  # force flush to disk
                self.logger.info("Processed batch %d/%d",
                                (i // batch_size) + 1,
                                (num_docs + batch_size - 1) // batch_size)

            self.logger.info("Saved embeddings to shelve db %s", index_path)

    def exact_knn_from_pckl(self, query, prompt):
        #todo, don't forget prompt!

class HuggingFaceEmbedderSentenceTransformers(BaseEmbedder):

    def __init__(self, model_name='', matryoshka_dim=None, normalize_embeddings=True):
        super().__init__(normalize_embeddings)

        # e.g. model_name = Qwen/Qwen3-Embedding-8B
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
            truncate_dim=matryoshka_dim,
        )
        self.logger.info("Primary model device: %s", next(self.model[0].auto_model.parameters()).device)
        self.logger.info("Matryoshka dimension set to: %s", matryoshka_dim if matryoshka_dim else "full")

    def embed_documents_batch(self, texts: list[str]) -> Tensor:
        embeddings_tensor = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False  # Since batching is handled at higher level
        )
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

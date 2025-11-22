from abc import ABC
from nl_pe.utils.setup_logging import setup_logging
import os
import torch
import gpytorch
import pickle
import faiss
import numpy as np
import time

class BaseActiveLearner(ABC):

    def __init__(self, config):

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))
        self.n_obs_iterations = self.config.get('active_learning', {}).get('n_obs_iterations')

    def get_single_rel_judgment(self, state, doc_id):
        if not hasattr(self, 'qrels_map'):
            qrels_path = self.config['data'].get('qrels_path')
            if not qrels_path:
                self.logger.error("Qrels path not specified in data config")
                raise ValueError("Qrels path not specified in data config")
            self.qrels_map = {}
            with open(qrels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        qid, _, pid, rel = parts[0], parts[1], parts[2], parts[3]
                        rel = float(rel)
                        if qid not in self.qrels_map:
                            self.qrels_map[qid] = {}
                        self.qrels_map[qid][pid] = rel
            self.logger.debug(f"Loaded qrels for {len(self.qrels_map)} queries")
        qid = state['qid']
        return self.qrels_map.get(qid, {}).get(doc_id, 0)

    def final_ranked_list_from_posterior(self, state):
        posterior_means = state["posterior_means"][-1]
        sorted_indices = sorted(range(len(posterior_means)), key=lambda i: posterior_means[i], reverse=True)
        doc_ids = state["doc_ids"]
        state["final_ranked_list"] = [doc_ids[i] for i in sorted_indices]

class GPActiveLearner(BaseActiveLearner):

    def __init__(self, config):
        super().__init__(config)

    def active_learn(self, state):
        # Load data
        index_path = self.config['data']['index_path']
        doc_ids_path = self.config['data']['doc_ids_path']
        all_embeddings = torch.load(index_path)
        doc_ids = pickle.load(open(doc_ids_path, 'rb'))
        state["doc_ids"] = doc_ids
        
        # GP config
        gp_config = self.config['gp']
        kernel = gp_config['kernel']
        lengthscale = gp_config['lengthscale']
        signal_noise = gp_config['signal_noise']
        observation_noise = gp_config['observation_noise']
        query_rel_label = gp_config['query_rel_label']
        
        # Active learning config
        acq_func_name = self.config['active_learning']['acquisition_f']
        
        # Initialize lists
        state["selected_doc_ids"] = []
        state["acquisition_scores"] = []
        state["acquisition_times"] = []
        state["posterior_means"] = []
        state["posterior_variances"] = []
        
        # First observation: query_embedding and its label
        X_obs = state["query_emb"].unsqueeze(0)
        y_obs = torch.tensor([query_rel_label], dtype=torch.float32)
        
        # Iterate
        for iteration in range(self.n_obs_iterations):
            # Create GP model
            class ExactGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super().__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                    self.covar_module.base_kernel.lengthscale = lengthscale
                    self.covar_module.outputscale = signal_noise
                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise = observation_noise
            model = ExactGPModel(X_obs, y_obs, likelihood)
            
            # Get acquisition start time
            acq_start = time.time()
            # Get acquisition scores for all docs except observed
            unobserved_indices = [i for i in range(len(doc_ids)) if doc_ids[i] not in state["selected_doc_ids"]]
            acq_scores = self.compute_acquisition_scores(model, all_embeddings, unobserved_indices, acq_func_name)
            acq_time = time.time() - acq_start
            
            # Select next doc
            best_idx_in_unobs = torch.argmax(acq_scores).item()
            selected_idx = unobserved_indices[best_idx_in_unobs]
            selected_doc_id = doc_ids[selected_idx]
            acq_score = acq_scores[best_idx_in_unobs].item()
            
            # Record
            state["selected_doc_ids"].append(selected_doc_id)
            state["acquisition_scores"].append(acq_score)
            state["acquisition_times"].append(acq_time)
            
            # Get label for selected doc
            y_new = self.get_single_rel_judgment(state, selected_doc_id)
            
            # Update observations
            X_new = all_embeddings[selected_idx].unsqueeze(0)
            X_obs = torch.cat([X_obs, X_new], dim=0)
            y_obs = torch.cat([y_obs, torch.tensor([y_new], dtype=torch.float32)], dim=0)
            
            # Record posteriors
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = likelihood(model(all_embeddings))
            state["posterior_means"].append(pred.mean.tolist())
            state["posterior_variances"].append(pred.variance.tolist())

    def compute_acquisition_scores(self, model, all_embeddings, unobserved_indices, acq_func_name):
        if acq_func_name == 'ts':
            return self.ts(model, all_embeddings, unobserved_indices)
        elif acq_func_name == 'ucb':
            return self.ucb(model, all_embeddings, unobserved_indices)
        elif acq_func_name == 'greedy':
            return self.greedy(model, all_embeddings, unobserved_indices)
        elif acq_func_name == 'greedy_epsilon':
            return self.greedy_epsilon(model, all_embeddings, unobserved_indices)
        elif acq_func_name == 'random':
            return self.random(all_embeddings, unobserved_indices)

    def ts(self, model, all_embeddings, unobserved_indices):
        unobserved_embs = all_embeddings[unobserved_indices]
        with torch.no_grad():
            pred = model(unobserved_embs)
            samples = pred.sample()
        return samples
    def ucb(self, model, all_embeddings, unobserved_indices):
        unobserved_embs = all_embeddings[unobserved_indices]
        with torch.no_grad():
            pred = model(unobserved_embs)
            beta = 2.0  # fixed or 2 * torch.log(torch.tensor(len(unobserved_indices) + 1))
            scores = pred.mean + beta * pred.stddev
        return scores

    def greedy(self, model, all_embeddings, unobserved_indices):
        unobserved_embs = all_embeddings[unobserved_indices]
        with torch.no_grad():
            pred = model(unobserved_embs)
            scores = pred.mean
        return scores

    def greedy_epsilon(self, model, all_embeddings, unobserved_indices, epsilon=0.1):
        scores = self.greedy(model, all_embeddings, unobserved_indices)
        n = scores.numel()
        rand_indices = torch.rand(n) < epsilon
        scores[rand_indices] = torch.randn(scores[rand_indices].size()).to(scores.device)
        return scores

    def random(self, all_embeddings, unobserved_indices):
        n = len(unobserved_indices)
        scores = torch.randn(n)
        return scores

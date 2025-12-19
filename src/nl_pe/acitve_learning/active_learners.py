from abc import ABC
from contextlib import nullcontext
from nl_pe.utils.setup_logging import setup_logging
import os
import torch
import gpytorch
import pickle
import faiss
import numpy as np
import math
import time

class BaseActiveLearner(ABC):

    def __init__(self, config):

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")
        self.n_obs_iterations = self.config.get('active_learning', {}).get('n_obs_iterations')

    def get_single_rel_judgment(self, state, doc_id):
        self.logger.debug(f"Getting relevance judgment for doc_id {doc_id} with qid {state.get('qid', 'unknown')}")
        if not hasattr(self, 'qrels_map'):
            data_config = self.config.get('data', {})
            qrels_path = data_config.get('qrels_path')
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
        judgment = self.qrels_map.get(qid, {}).get(doc_id, 0)
        self.logger.debug(f"Relevance judgment for doc_id {doc_id} is {judgment}")
        return judgment


class GPActiveLearner(BaseActiveLearner):

    def __init__(self, config):
        super().__init__(config)

        # GP config
        self.gp_config = self.config.get('gp', {})

        fast_pred = self.gp_config.get("fast_pred", False)
        self.fast_ctx = gpytorch.settings.fast_pred_var() if fast_pred else nullcontext()
        if fast_pred:
            self.logger.info("Using fast_pred_var")

    def _maybe_refit_gp(self, state, model, likelihood, train_x, train_y):

        refit_after_obs = self.gp_config.get('refit_after_obs')
        k_refit = int(self.gp_config.get('k_refit') or 0)
        lr = self.gp_config.get('lr')
        k_obs_refit = int(self.gp_config.get('k_obs_refit') or 1)

        # Only refit if requested and k_refit > 0
        if str(refit_after_obs).lower() not in ("1", "true", "y", "yes", "true"):
            return
        if k_refit is None or k_refit <= 0:
            return

        self.logger.debug(f"Refitting GP hyperparameters for {k_refit} steps")
        model.train()
        likelihood.train()

        # Only refit every k_obs_refit observations
        obs_count = train_x.size(0)
        if k_obs_refit is not None and k_obs_refit > 1 and (obs_count % k_obs_refit != 0):
            return

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=lr,
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for step in range(k_refit):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            neg_mll = loss.item()

            self.logger.debug(f"Refit step {step + 1}/{k_refit}, -mll={neg_mll:.6f}")
            ls = float(model.covar_module.base_kernel.lengthscale.item())
            sn = float(model.covar_module.outputscale.item())
            on = float(likelihood.noise.item())
            state["neg_mll"].append(neg_mll)
            state["lengthscale"].append(ls)
            state["signal_noise"].append(sn)
            state["obs_noise"].append(on)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

    def active_learn(self, state):
        self.logger.debug("Starting active_learn")
        # Load data
        data_config = self.config.get('data', {})
        index_path = data_config.get('index_path')
        doc_ids_path = data_config.get('doc_ids_path')
        if not index_path or not doc_ids_path:
            raise ValueError("Index path and doc_ids_path must be specified in data config")
        index = faiss.read_index(index_path)
        #todo: don't load all embeddings if too large
        xb_np = index.reconstruct_n(0, index.ntotal)
        all_embeddings = torch.from_numpy(xb_np).float()
        doc_ids = pickle.load(open(doc_ids_path, 'rb'))
        self.logger.debug(f"Loaded {len(doc_ids)} documents and embeddings with shape {all_embeddings.shape}")
        
        #todo: use other kernels if needed
        kernel = self.gp_config.get('kernel', 'rbf')  # 'rbf' is standard, can keep or remove
        lengthscale = self.gp_config.get('lengthscale')
        signal_noise = self.gp_config.get('signal_noise')
        observation_noise = self.gp_config.get('observation_noise')
        query_rel_label = self.gp_config.get('query_rel_label')
        k_final = int(self.gp_config.get('k_final'))

        use_query_reforms = str(self.gp_config.get('use_query_reformulations', False)).lower() in ("1", "true", "yes", "y")
        reform_query_rel_label = self.gp_config.get('reform_query_rel_label')

        #warm start percent: none or 0 to 100      
        warm_start_percent = float(self.gp_config.get('warm_start_percent', 0))

        # Active learning config
        self.al_config = self.config.get('active_learning', {})
        acq_func_name = self.al_config.get('acquisition_f')
        
        # Initialize lists
        state["selected_doc_ids"] = []
        state["acquisition_scores"] = []
        state["acquisition_times"] = []
        state["model_update_times"] = []
        state["neg_mll"] = []
        state["lengthscale"] = []
        state["signal_noise"] = []
        state["obs_noise"] = []
        
        # First observation: query_embedding and its label
        X_obs = state["query_emb"].unsqueeze(0)
        y_obs = torch.tensor([query_rel_label], dtype=torch.float32)
        self.logger.debug(f"First observation set with label {query_rel_label}")

        if use_query_reforms:
            reform_embs = state.get("query_reformation_embeddings", None)

            # reform_embs is expected to be a 2D tensor: (n_reforms, d)
            if isinstance(reform_embs, torch.Tensor) and reform_embs.numel() > 0 and reform_query_rel_label is not None:
                n_reforms = reform_embs.size(0)
                reform_y = torch.full(
                    (n_reforms,),
                    float(reform_query_rel_label),
                    dtype=torch.float32,
                )
                X_obs = torch.cat([X_obs, reform_embs], dim=0)
                y_obs = torch.cat([y_obs, reform_y], dim=0)
                self.logger.debug(
                    f"Added {n_reforms} query reformulation embeddings "
                    f"with label {reform_query_rel_label} to initial observations"
                )
            else:
                self.logger.debug(
                    "use_query_reformulations=True but no valid reformulation embeddings "
                    "or reform_query_rel_label is None; skipping reformulations."
                )

    
        # Number of active learning iterations (may be reduced by warm start)
        n_iterations = self.n_obs_iterations

        # Warm start observations
        if warm_start_percent > 0:
            top_k_psgs = state.get('top_k_psgs', [])
            if not top_k_psgs:
                raise ValueError("Warm start requested but 'top_k_psgs' not found in state")
            else:
                n_candidates = len(top_k_psgs)
                if warm_start_percent >= 100.0:
                    n_warm = n_candidates
                else:
                    n_warm = int(np.floor(n_candidates * (warm_start_percent / 100.0)))
                    if n_warm <= 0:
                        n_warm = 1  # at least one warm-start doc if percent > 0

                n_warm = min(n_warm, n_candidates)
                warm_start_doc_ids = top_k_psgs[:n_warm]

                self.logger.info(
                    f"Warm start enabled: percent={warm_start_percent}, "
                    f"n_candidates={n_candidates}, n_warm={n_warm}"
                )

                # Map doc_id -> index in doc_ids
                docid_to_idx = {d_id: i for i, d_id in enumerate(doc_ids)}

                warm_added = 0
                for d_id in warm_start_doc_ids:
                    idx = docid_to_idx.get(d_id, None)
                    if idx is None:
                        self.logger.warning(
                            f"Warm start doc_id {d_id} not found in loaded doc_ids; skipping."
                        )
                        continue

                    # Get label and embedding
                    y_new = self.get_single_rel_judgment(state, d_id)
                    X_new = all_embeddings[idx].unsqueeze(0)

                    # Update observations and selected docs
                    X_obs = torch.cat([X_obs, X_new], dim=0)
                    y_obs = torch.cat([y_obs, torch.tensor([y_new], dtype=torch.float32)], dim=0)
                    state["selected_doc_ids"].append(d_id)
                    warm_added += 1

                # Reduce the number of AL iterations by the number of warm-start observations
                if warm_added > 0:
                    n_iterations = max(0, self.n_obs_iterations - warm_added)
                    self.logger.debug(
                        f"Warm start added {warm_added} observations; "
                        f"active learning iterations reduced from "
                        f"{self.n_obs_iterations} to {n_iterations}"
                    )
                else:
                    self.logger.debug(
                        "No warm start observations were actually added; "
                        "keeping original number of active learning iterations."
                    )

                self.logger.info(f"Building initial GP model after warm start with {len(X_obs)} observations")
                model_build_start = time.time()
                # Create GP model
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihood.noise = observation_noise
                model = ExactGPModel(X_obs, y_obs, likelihood, lengthscale, signal_noise)
                self.logger.debug("GP model created")

                # Optionally refit GP hyperparameters on all observed data
                self._maybe_refit_gp(state, model, likelihood, X_obs, y_obs)
                model_build_time = time.time() - model_build_start
                state["model_update_times"].append(model_build_time)
                self.logger.debug(f"Model update (build + optional refit) took {model_build_time:.2f} seconds")

                model.eval()
                likelihood.eval()

        # Iterate
        for iteration in range(n_iterations):
            self.logger.debug(f"Active learning iteration {iteration + 1}/{n_iterations}")
            model_build_start = time.time()
            # Create GP model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise = observation_noise
            model = ExactGPModel(X_obs, y_obs, likelihood, lengthscale, signal_noise)
            self.logger.debug("GP model created for this iteration")

            # Optionally refit GP hyperparameters on all observed data
            self._maybe_refit_gp(state, model, likelihood, X_obs, y_obs)
            model_build_time = time.time() - model_build_start
            state["model_update_times"].append(model_build_time)
            self.logger.debug(f"Model update (build + optional refit) took {model_build_time:.2f} seconds")

            model.eval()
            likelihood.eval()

            # Get acquisition start time
            acq_start = time.time()
            # Get acquisition scores for all docs except observed
            unobserved_indices = [i for i in range(len(doc_ids)) if doc_ids[i] not in state["selected_doc_ids"]]
            self.logger.debug(f"Computing acquisition scores for {len(unobserved_indices)} unobserved documents")
            acq_scores = self.compute_acquisition_scores(model, all_embeddings, unobserved_indices, acq_func_name)
            acq_time = time.time() - acq_start
            self.logger.debug(f"Acquisition scores computed in {acq_time:.2f} seconds, max score: {torch.max(acq_scores).item():.4f}")
            
            # Select next doc
            best_idx_in_unobs = torch.argmax(acq_scores).item()
            selected_idx = unobserved_indices[best_idx_in_unobs]
            selected_doc_id = doc_ids[selected_idx]
            acq_score = acq_scores[best_idx_in_unobs].item()
            self.logger.debug(f"Selected document {selected_doc_id} with acquisition score {acq_score:.4f}")

            # Record
            state["selected_doc_ids"].append(selected_doc_id)
            state["acquisition_scores"].append(acq_score)
            state["acquisition_times"].append(acq_time)

            # Get label for selected doc
            y_new = self.get_single_rel_judgment(state, selected_doc_id)
            self.logger.debug(f"Retrieved relevance label {y_new} for document {selected_doc_id}")

            # Update observations
            X_new = all_embeddings[selected_idx].unsqueeze(0)
            X_obs = torch.cat([X_obs, X_new], dim=0)
            y_obs = torch.cat([y_obs, torch.tensor([y_new], dtype=torch.float32)], dim=0)
            self.logger.debug(f"Observations updated to {len(X_obs)} points")

        # Final ranked list
        model.eval()
        likelihood.eval()
        pred = likelihood(model(all_embeddings))

        self.logger.debug("Creating final ranked list from posterior means")
        posterior_means = pred.mean.tolist()
        sorted_indices = sorted(range(len(posterior_means)), key=lambda i: posterior_means[i], reverse=True)
        state["top_k_psgs"] = [doc_ids[i] for i in sorted_indices[:k_final]]

        # pop embeddings
        if "query_emb" in state:
            state.pop("query_emb")
        if "query_reformation_embeddings" in state:
            state.pop("query_reformation_embeddings")

        self.logger.debug(f"Final ranked list created with top 5 docs: {state['top_k_psgs'][:5]}")

        
    def compute_acquisition_scores(self, model, all_embeddings, unobserved_indices, acq_func_name):
        self.logger.debug(f"Computing acquisition scores using '{acq_func_name}' for {len(unobserved_indices)} unobserved documents")
        if acq_func_name == 'ts':
            return self.ts(model, all_embeddings, unobserved_indices)
        elif acq_func_name == 'ucb_const_beta':
            return self.ucb_const_beta(model, all_embeddings, unobserved_indices)
        elif acq_func_name == 'greedy':
            return self.greedy(model, all_embeddings, unobserved_indices)
        elif acq_func_name == 'greedy_epsilon':
            return self.greedy_epsilon(model, all_embeddings, unobserved_indices)
        elif acq_func_name == 'random':
            return self.random(all_embeddings, unobserved_indices)
        elif acq_func_name == 'greedy_epsilon_ts':
            return self.greedy_epsilon_ts(model, all_embeddings, unobserved_indices)

    def ts(self, model, all_embeddings, unobserved_indices):
        self.logger.debug("Acquiring scores via Thompson Sampling: sampling from posterior")
        unobserved_embs = all_embeddings[unobserved_indices]
        with torch.no_grad(), self.fast_ctx:
            pred = model(unobserved_embs)
            samples = pred.sample()
        return samples
    
    def greedy_epsilon_ts(self, model, all_embeddings, unobserved_indices):
        """
        With probability (1 - epsilon), return greedy (posterior mean) scores.
        With probability epsilon, return Thompson Sampling scores.

        This is an epsilon-greedy variant where the exploration strategy
        is Thompson Sampling rather than uniform random.
        """
        epsilon = self.config.get('active_learning', {}).get('epsilon')

        if torch.rand(1).item() > epsilon:
            # Exploit: greedy by posterior mean
            self.logger.debug(
                f"Greedy-epsilon-TS: taking GREEDY action (1-epsilon={1 - epsilon:.3f})"
            )
            return self.greedy(model, all_embeddings, unobserved_indices)
        else:
            # Explore: Thompson sample from posterior
            self.logger.debug(
                f"Greedy-epsilon-TS: taking TS action (epsilon={epsilon:.3f})"
            )
            return self.ts(model, all_embeddings, unobserved_indices)


    def ucb_const_beta(self, model, all_embeddings, unobserved_indices):
        if 'ucb_beta_const' not in self.al_config:
            raise KeyError(
                "UCB acquisition requires 'ucb_beta_const' in config['active_learning']"
            )
        beta = float(self.al_config['ucb_beta_const'])
        sqrt_beta = math.sqrt(beta)

        self.logger.debug(f"Acquiring scores via UCB with beta={beta}")

        unobserved_embs = all_embeddings[unobserved_indices]

        with torch.no_grad(), self.fast_ctx:
            pred = model(unobserved_embs)
            scores = pred.mean + sqrt_beta * pred.stddev

        return scores


    def greedy(self, model, all_embeddings, unobserved_indices):
        self.logger.debug("Acquiring scores via greedy: using posterior means")
        unobserved_embs = all_embeddings[unobserved_indices]
        with torch.no_grad(), self.fast_ctx:
            pred = model(unobserved_embs)
            scores = pred.mean
        return scores

    def greedy_epsilon(self, model, all_embeddings, unobserved_indices):
        """
        With probability (1 - epsilon), return greedy scores.
        With probability epsilon, return random scores (so that the selected
        action is random when argmax is taken).
        """
        epsilon = self.config.get('active_learning', {}).get('epsilon')

        if torch.rand(1).item() > epsilon:
            # Greedy case
            self.logger.debug(f"Greedy-epsilon: taking GREEDY action (1-epsilon={1-epsilon})")
            return self.greedy(model, all_embeddings, unobserved_indices)
        else:
            # Random case
            self.logger.debug(f"Greedy-epsilon: taking RANDOM action (epsilon={epsilon})")
            n = len(unobserved_indices)
            # Uniform random scores ensures argmax is random
            return torch.randn(n, device=all_embeddings.device)


    def random(self, all_embeddings, unobserved_indices):
        self.logger.debug("Acquiring scores randomly (baseline)")
        n = len(unobserved_indices)
        scores = torch.randn(n)
        return scores


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale, signal_noise):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = lengthscale
        self.covar_module.outputscale = signal_noise
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
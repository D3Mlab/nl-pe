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

        # Set device
        tensor_ops_device = self.config.get('tensor_ops_device', 'cpu')
        self.device = torch.device('cuda' if tensor_ops_device == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

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
        qid = str(state['qid'])
        judgment = self.qrels_map.get(qid, {}).get(doc_id, 0)
        self.logger.debug(f"Relevance judgment for doc_id {doc_id} is {judgment}")
        return judgment


class GPActiveLearner(BaseActiveLearner):

    def __init__(self, config):
        super().__init__(config)

        # Data config for index and batch size
        data_config = self.config.get('data', {})
        index_path = data_config.get('index_path')
        self.index = faiss.read_index(index_path)
        doc_ids_path = data_config.get('doc_ids_path')
        self.doc_ids = pickle.load(open(doc_ids_path, 'rb'))
        self.embedding_batch_size = data_config.get('embedding_batch_size', len(self.doc_ids))

        # GP config
        self.gp_config = self.config.get('gp', {})

        #learning config
        self.opt_config = self.config.get('optimization', {})

        fast_pred = self.gp_config.get("fast_pred", False)
        self.fast_ctx = gpytorch.settings.fast_pred_var() if fast_pred else nullcontext()
        if fast_pred:
            self.logger.info("Using fast_pred_var")

    def _maybe_refit_gp(self, state, model, likelihood, train_x, train_y):

        refit_after_obs = self.opt_config.get('refit_after_obs')
        k_refit = int(self.opt_config.get('k_refit') or 0)
        lr = self.opt_config.get('lr')
        k_obs_refit = int(self.opt_config.get('k_obs_refit') or 1)
        opt_noise = bool(self.opt_config.get("opt_noise", True))
        opt_sig_noise = bool(self.opt_config.get("opt_sig_noise", True))


        # Only refit if requested and k_refit > 0
        if str(refit_after_obs).lower() not in ("1", "true", "y", "yes", "true"):
            return
        if k_refit is None or k_refit <= 0:
            return

        with torch.set_grad_enabled(True):
            # ensure train tensors are real autograd tensors
            train_x = train_x.clone()
            train_y = train_y.clone()
            
            self.logger.debug(f"Refitting GP hyperparameters for {k_refit} steps")
            model.train()
            likelihood.train()

            # Only refit every k_obs_refit observations
            obs_count = train_x.size(0)
            if k_obs_refit is not None and k_obs_refit > 1 and (obs_count % k_obs_refit != 0):
                return

            params = []

            # optionally optimize outputscale (signal variance)
            if opt_sig_noise:
                params += list(model.covar_module.parameters())  # includes outputscale
            else:
                model.covar_module.raw_outputscale.requires_grad_(False)
                # always optimize kernel lengthscales
                params += list(model.covar_module.base_kernel.parameters())
            # optionally optimize observation noise
            if opt_noise:
                params += list(likelihood.parameters())
            else:
                likelihood.raw_noise.requires_grad_(False)
            optimizer = torch.optim.Adam(params, lr=lr)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for step in range(k_refit):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                neg_mll = loss.item()
                state["neg_mll"].append(neg_mll)
                self.logger.debug(f"Refit step {step + 1}/{k_refit}, -mll={neg_mll:.6f}")
                loss.backward()
                optimizer.step()

            model.eval()
            likelihood.eval()

        # record only the final values after refit
        with torch.no_grad():
            ls_t = model.covar_module.base_kernel.lengthscale.detach().cpu()
            if ls_t.numel() == 1:
                ls = float(ls_t.item())
            else:
                ls = ls_t.squeeze().tolist()

            sn = float(model.covar_module.outputscale.item())
            on = float(likelihood.noise.item())

        #state["neg_mll"].append(neg_mll)
        state["lengthscale"].append(ls)
        state["signal_noise"].append(sn)
        state["obs_noise"].append(on)


    def active_learn(self, state):
        # Data already loaded in __init__
        self.logger.info(f"Using {len(self.doc_ids)} documents, batch_size={self.embedding_batch_size}")
        
        #todo: use other kernels if needed
        kernel = self.gp_config.get('kernel', 'rbf')  # 'rbf' is standard, can keep or remove
        lengthscale = self.gp_config.get('lengthscale')
        signal_noise = self.gp_config.get('signal_noise')
        observation_noise = self.gp_config.get('observation_noise')
        query_rel_label = self.gp_config.get('query_rel_label')
        k_final = int(self.gp_config.get('k_final'))


        #warm start percent: none or 0 to 100      
        warm_start_percent = float(self.gp_config.get('warm_start_percent', 0))

        # Active learning config
        self.al_config = self.config.get('active_learning', {})
        acq_func_name = self.al_config.get('acquisition_f')
        
        #optimization
        self.ard = self.opt_config.get('ard')

        # Initialize lists
        state["selected_doc_ids"] = []
        state["acquisition_scores"] = []
        state["acquisition_times"] = []
        state["acquisition_IO_times"] = []
        state["model_update_times"] = []
        state["neg_mll"] = []
        state["lengthscale"] = []
        state["signal_noise"] = []
        state["obs_noise"] = []
        
        # First observation: query_embedding and its label
        X_obs = state["query_emb"].unsqueeze(0).to(self.device)
        y_obs = torch.tensor([query_rel_label], dtype=torch.float32).to(self.device)
        self.logger.debug(f"First observation set with label {query_rel_label}")

    
        # Warm start observations
        remaining_obs_post_ws = self.n_obs_iterations
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
                docid_to_idx = {d_id: i for i, d_id in enumerate(self.doc_ids)}

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
                    X_new = torch.from_numpy(self.index.reconstruct(idx)).float().unsqueeze(0).to(self.device)

                    # Update observations and selected docs
                    X_obs = torch.cat([X_obs, X_new], dim=0)
                    y_obs = torch.cat([y_obs, torch.tensor([y_new], dtype=torch.float32).to(self.device)], dim=0)
                    state["selected_doc_ids"].append(d_id)
                    warm_added += 1

                # Reduce the number of AL iterations by the number of warm-start observations
                if warm_added > 0:
                    remaining_obs_post_ws = max(0, self.n_obs_iterations - warm_added)
                    self.logger.debug(
                        f"Warm start added {warm_added} observations; "
                        f"active learning iterations reduced from "
                        f"{self.n_obs_iterations} to {remaining_obs_post_ws}"
                    )
                else:
                    self.logger.debug(
                        "No warm start observations were actually added; "
                        "keeping original number of active learning iterations."
                    )

        # BO iterations
        for iteration in range(remaining_obs_post_ws):
            self.logger.debug(f"Active learning iteration {iteration + 1}/{remaining_obs_post_ws}")

            model, likelihood = self._build_and_maybe_refit_gp(
                state,
                X_obs,
                y_obs,
                lengthscale=lengthscale,
                signal_noise=signal_noise,
                observation_noise=observation_noise,
            )

            # Get acquisition scores for all docs except observed
            unobserved_indices = [i for i in range(len(self.doc_ids)) if self.doc_ids[i] not in state["selected_doc_ids"]]
            self.logger.debug(f"Computing acquisition scores for {len(unobserved_indices)} unobserved documents")
            best_idx_in_unobs, acq_score, acq_gp_time, acq_io_time = self.compute_acquisition_scores(model, unobserved_indices, acq_func_name)

            # Select next doc
            selected_idx = unobserved_indices[best_idx_in_unobs]
            selected_doc_id = self.doc_ids[selected_idx]
            self.logger.debug(f"Selected document {selected_doc_id} with acquisition score {acq_score:.4f}")

            # Record
            state["selected_doc_ids"].append(selected_doc_id)
            state["acquisition_scores"].append(acq_score)
            state["acquisition_times"].append(acq_gp_time)
            state["acquisition_IO_times"].append(acq_io_time)

            # Get label for selected doc
            y_new = self.get_single_rel_judgment(state, selected_doc_id)
            self.logger.debug(f"Retrieved relevance label {y_new} for document {selected_doc_id}")

            # Update observations
            X_new = torch.from_numpy(self.index.reconstruct(selected_idx)).float().unsqueeze(0).to(self.device)
            X_obs = torch.cat([X_obs, X_new], dim=0)
            y_obs = torch.cat([y_obs, torch.tensor([y_new], dtype=torch.float32).to(self.device)], dim=0)
            self.logger.debug(f"Observations updated to {len(X_obs)} points")

        # Final model after all observations
        model, likelihood = self._build_and_maybe_refit_gp(
            state,
            X_obs,
            y_obs,
            lengthscale=lengthscale,
            signal_noise=signal_noise,
            observation_noise=observation_noise,
        )

        n_total = self.index.ntotal
        batch_size = self.embedding_batch_size
        n_batches = math.ceil(n_total / batch_size)
        posterior_means = []
        final_gp_time = 0.0
        final_io_time = 0.0
        with torch.no_grad(), self.fast_ctx:
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_total)
                io_start = time.time()
                batch_embs = torch.from_numpy(self.index.reconstruct_n(start, end - start)).float().to(self.device)
                io_time = time.time() - io_start
                final_io_time += io_time
                gp_start = time.time()
                pred_batch = model(batch_embs)
                gp_time = time.time() - gp_start
                final_gp_time += gp_time
                posterior_means.extend(pred_batch.mean.tolist())
                del batch_embs, pred_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        state["final_inf_time"] = final_gp_time
        state["final_IO_time"] = final_io_time
        self.logger.debug("Creating final ranked list from posterior means")
        sorted_indices = sorted(range(len(posterior_means)), key=lambda i: posterior_means[i], reverse=True)
        state["top_k_psgs"] = [self.doc_ids[i] for i in sorted_indices[:k_final]]

        # pop embeddings
        if "query_emb" in state:
            state.pop("query_emb")

        self.logger.debug(f"Final ranked list created with top 5 docs: {state['top_k_psgs'][:5]}")

        
    def compute_acquisition_scores(self, model, unobserved_indices, acq_func_name):
        self.logger.debug(f"Computing acquisition scores using '{acq_func_name}' for {len(unobserved_indices)} unobserved documents")
        n_unobs = len(unobserved_indices)
        batch_size = self.embedding_batch_size
        best_score = float('-inf')
        best_idx_in_unobs = -1
        total_io_time = 0.0
        total_gp_time = 0.0

        if acq_func_name == 'random':
            # Special case, no embeddings needed
            best_idx_in_unobs = torch.randint(0, n_unobs, (1,)).item()
            best_score = 0.0  # dummy
            return best_idx_in_unobs, best_score, 0.0, 0.0

        # For other methods, batch process
        with torch.no_grad(), self.fast_ctx:
            for i in range(0, n_unobs, batch_size):
                end = min(i + batch_size, n_unobs)
                batch_indices = unobserved_indices[i:end]

                # IO time: reconstructing embeddings
                io_start = time.time()
                batch_embs_list = []
                for idx in batch_indices:
                    emb = self.index.reconstruct(idx)
                    batch_embs_list.append(emb)
                batch_embs_np = np.array(batch_embs_list)
                batch_embs = torch.from_numpy(batch_embs_np).float().to(self.device)
                io_time = time.time() - io_start
                total_io_time += io_time

                # GP time: model predictions
                gp_start = time.time()
                if acq_func_name == 'ts':
                    batch_max_score, batch_max_idx = self._ts_batch(model, batch_embs)
                elif acq_func_name == 'ucb_const_beta':
                    batch_max_score, batch_max_idx = self._ucb_batch(model, batch_embs)
                elif acq_func_name == 'greedy':
                    batch_max_score, batch_max_idx = self._greedy_batch(model, batch_embs)
                elif acq_func_name == 'greedy_epsilon':
                    batch_max_score, batch_max_idx = self._greedy_epsilon_batch(model, batch_embs)
                elif acq_func_name == 'greedy_epsilon_ts':
                    batch_max_score, batch_max_idx = self._greedy_epsilon_ts_batch(model, batch_embs)
                elif acq_func_name == 'lse_straddle':
                    batch_max_score, batch_max_idx = self._lse_straddle_batch(model, batch_embs)
                elif acq_func_name == 'lse_margin':
                    batch_max_score, batch_max_idx = self._lse_margin_batch(model, batch_embs)

                else:
                    raise ValueError(f"Unknown acquisition function: {acq_func_name}")
                gp_time = time.time() - gp_start
                total_gp_time += gp_time

                if batch_max_score > best_score:
                    best_score = batch_max_score
                    best_idx_in_unobs = i + batch_max_idx

                del batch_embs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return best_idx_in_unobs, best_score, total_gp_time, total_io_time

    def _ts_batch(self, model, batch_embs):
        pred = model(batch_embs)
        samples = pred.sample()
        max_score, max_idx = torch.max(samples, dim=0)
        return max_score.item(), max_idx.item()

    def _ucb_batch(self, model, batch_embs):
        if 'ucb_beta_const' not in self.al_config:
            raise KeyError("UCB acquisition requires 'ucb_beta_const' in config['active_learning']")
        beta = float(self.al_config['ucb_beta_const'])
        sqrt_beta = math.sqrt(beta)
        pred = model(batch_embs)
        scores = pred.mean + sqrt_beta * pred.stddev
        max_score, max_idx = torch.max(scores, dim=0)
        return max_score.item(), max_idx.item()

    def _greedy_batch(self, model, batch_embs):
        pred = model(batch_embs)
        scores = pred.mean
        max_score, max_idx = torch.max(scores, dim=0)
        return max_score.item(), max_idx.item()

    def _greedy_epsilon_batch(self, model, batch_embs):
        epsilon = self.config.get('active_learning', {}).get('epsilon')
        if torch.rand(1).item() > epsilon:
            return self._greedy_batch(model, batch_embs)
        else:
            batch_size = batch_embs.size(0)
            random_idx = torch.randint(0, batch_size, (1,)).item()
            return float('-inf'), random_idx  # low score so not selected

    def _greedy_epsilon_ts_batch(self, model, batch_embs):
        epsilon = self.config.get('active_learning', {}).get('epsilon')
        if torch.rand(1).item() > epsilon:
            return self._greedy_batch(model, batch_embs)
        else:
            return self._ts_batch(model, batch_embs)

    def _lse_straddle_batch(self, model, batch_embs):
        tau = float(self.al_config.get("lse_tau"))
        kappa = float(self.al_config.get("lse_kappa"))

        pred = model(batch_embs)
        mu = pred.mean
        sigma = pred.stddev

        scores = -torch.abs(mu - tau) + kappa * sigma

        max_score, max_idx = torch.max(scores, dim=0)
        return max_score.item(), max_idx.item()

    def _lse_margin_batch(self, model, batch_embs):
        tau = float(self.al_config.get("lse_tau"))

        pred = model(batch_embs)
        mu = pred.mean
        sigma = pred.stddev

        scores = -torch.abs(mu - tau) / (sigma + 1e-8)

        max_score, max_idx = torch.max(scores, dim=0)
        return max_score.item(), max_idx.item()


    def _build_and_maybe_refit_gp(
        self,
        state,
        X_obs,
        y_obs,
        *,
        lengthscale,
        signal_noise,
        observation_noise,
    ):
        start = time.time()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=observation_noise)
        likelihood = likelihood.to(self.device)

        model = ExactGPModel(
            X_obs,
            y_obs,
            likelihood,
            lengthscale,
            signal_noise,
            ard = self.ard
        ).to(self.device)

        self._maybe_refit_gp(state, model, likelihood, X_obs, y_obs)

        elapsed = time.time() - start
        state["model_update_times"].append(elapsed)

        model.eval()
        likelihood.eval()

        return model, likelihood

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale, signal_noise, ard=False):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()


        if ard:
            d = train_x.size(-1)
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=d)
        else:
            base_kernel = gpytorch.kernels.RBFKernel()

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        #assuming a scalar ls broadcast to d for now
        self.covar_module.base_kernel.initialize(lengthscale=lengthscale)
        self.covar_module.initialize(outputscale=signal_noise)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




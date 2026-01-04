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

from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy
)


# ================================================================
# Base class
# ================================================================
class BaseActiveLearner(ABC):

    def __init__(self, config):

        self.config = config
        self.logger = setup_logging(
            self.__class__.__name__,
            config=self.config,
            output_file=os.path.join(self.config['exp_dir'], "experiment.log")
        )
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")
        self.n_obs_iterations = self.config.get('active_learning', {}).get('n_obs_iterations')

        tensor_ops_device = self.config.get('tensor_ops_device', 'cpu')
        self.device = torch.device(
            'cuda' if tensor_ops_device == 'gpu' and torch.cuda.is_available() else 'cpu'
        )
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
                        qid, _, pid, rel = parts
                        rel = float(rel)
                        self.qrels_map.setdefault(qid, {})[pid] = rel

            self.logger.debug(f"Loaded qrels for {len(self.qrels_map)} queries")

        qid = str(state['qid'])
        judgment = self.qrels_map.get(qid, {}).get(doc_id, 0)
        self.logger.debug(f"Relevance judgment for doc_id {doc_id} is {judgment}")
        return judgment


# ================================================================
# GP Active Learner
# ================================================================
class GPActiveLearner(BaseActiveLearner):

    def __init__(self, config):
        super().__init__(config)

        data_config = self.config.get('data', {})
        index_path = data_config.get('index_path')
        self.index = faiss.read_index(index_path)

        doc_ids_path = data_config.get('doc_ids_path')
        self.doc_ids = pickle.load(open(doc_ids_path, 'rb'))
        self.embedding_batch_size = data_config.get(
            'embedding_batch_size',
            len(self.doc_ids)
        )

        self.gp_config = self.config.get('gp', {})
        self.opt_config = self.config.get('optimization', {})

        # =========================================================
        # task type
        # =========================================================
        self.gp_task_type = self.gp_config.get("task_type", "regression")

        # dirichlet + variational prior bias
        self.prior_bias_relevant = float(self.gp_config.get("prior_bias_relevant", 0.0))
        self.num_prob_samples = int(self.gp_config.get("num_prob_samples", 256))

        # variational params
        self.variational_training_iterations = int(
            self.gp_config.get("variational_training_iterations", 50)
        )
        self.variational_lr = float(
            self.gp_config.get("variational_lr", 0.05)
        )

        fast_pred = self.gp_config.get("fast_pred", False)
        self.fast_ctx = gpytorch.settings.fast_pred_var() if fast_pred else nullcontext()

        if fast_pred:
            self.logger.info("Using fast_pred_var")

        self.logger.info(f"GP task type = {self.gp_task_type}")

    # ================================================================
    # refit: exact GP case
    # ================================================================
    def _maybe_refit_gp(self, state, model, likelihood, train_x, train_y):

        refit_after_obs = self.opt_config.get('refit_after_obs')
        k_refit = int(self.opt_config.get('k_refit') or 0)
        lr = self.opt_config.get('lr')
        k_obs_refit = int(self.opt_config.get('k_obs_refit') or 1)
        opt_noise = bool(self.opt_config.get("opt_noise", True))
        opt_sig_noise = bool(self.opt_config.get("opt_sig_noise", True))

        if str(refit_after_obs).lower() not in ("1", "true", "y", "yes"):
            return
        if k_refit is None or k_refit <= 0:
            return

        with torch.set_grad_enabled(True):
            model.train()
            likelihood.train()

            obs_count = train_x.size(0)
            if k_obs_refit and k_obs_refit > 1 and (obs_count % k_obs_refit != 0):
                return

            params = []

            if opt_sig_noise:
                params += list(model.covar_module.parameters())
            else:
                model.covar_module.raw_outputscale.requires_grad_(False)
                params += list(model.covar_module.base_kernel.parameters())

            if opt_noise:
                params += list(likelihood.parameters())
            else:
                if hasattr(likelihood, "raw_noise"):
                    likelihood.raw_noise.requires_grad_(False)

            optimizer = torch.optim.Adam(params, lr=lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for step in range(k_refit):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                if loss.ndim > 0:
                    loss = loss.sum()
                loss.backward()
                optimizer.step()

            model.eval()
            likelihood.eval()

        with torch.no_grad():
            ls_t = model.covar_module.base_kernel.lengthscale.detach().cpu()
            ls = float(ls_t.item()) if ls_t.numel() == 1 else ls_t.squeeze().tolist()

            sn = float(model.covar_module.outputscale.item())

            if hasattr(likelihood, "noise"):
                on = float(likelihood.noise.item())
            elif hasattr(likelihood, "second_noise_covar"):
                on = float(likelihood.second_noise_covar.noise.mean().item())
            else:
                on = float("nan")

        state["lengthscale"].append(ls)
        state["signal_noise"].append(sn)
        state["obs_noise"].append(on)

    # ================================================================
    # VARIATIONAL TRAIN LOOP
    # ================================================================
    def _train_variational_gp(self, model, likelihood, train_x, labels):

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.variational_lr
        )

        mll = gpytorch.mlls.VariationalELBO(
            likelihood,
            model,
            labels.numel()
        )

        for i in range(self.variational_training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

    # ================================================================
    # BUILD GP (ALL MODES)
    # ================================================================
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

        # ---------------- VARIATIONAL ----------------
        if self.gp_task_type == "variational_binary":
            with torch.no_grad():
                labels = (y_obs > 0).long()

            likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)

            model = VariationalGPClassificationModel(
                X_obs,
                lengthscale,
                signal_noise,
                ard=self.ard,
            ).to(self.device)

            # prior bias here
            with torch.no_grad():
                model.mean_module.initialize(constant=self.prior_bias_relevant)

            self._train_variational_gp(model, likelihood, X_obs, labels)

        # ---------------- DIRICHLET ----------------
        elif self.gp_task_type == "binary_classification":
            with torch.no_grad():
                labels = (y_obs > 0).long()

            likelihood = DirichletClassificationLikelihood(
                labels,
                learn_additional_noise=True
            ).to(self.device)

            transformed_targets = likelihood.transformed_targets
            num_classes = likelihood.num_classes

            model = DirichletExactGPModel(
                X_obs,
                transformed_targets,
                likelihood,
                num_classes=num_classes,
                lengthscale=lengthscale,
                signal_noise=signal_noise,
                ard=self.ard,
            ).to(self.device)

            with torch.no_grad():
                bias = torch.zeros(num_classes, device=self.device)
                if num_classes >= 2:
                    bias[1] = self.prior_bias_relevant
                model.mean_module.initialize(constant=bias)

            self._maybe_refit_gp(state, model, likelihood, X_obs, transformed_targets)

        # ---------------- REGRESSION ----------------
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.initialize(noise=observation_noise)
            likelihood = likelihood.to(self.device)

            model = ExactGPModel(
                X_obs,
                y_obs,
                likelihood,
                lengthscale,
                signal_noise,
                ard=self.ard
            ).to(self.device)

            self._maybe_refit_gp(state, model, likelihood, X_obs, y_obs)

        elapsed = time.time() - start
        state["model_update_times"].append(elapsed)

        model.eval()
        likelihood.eval()

        return model, likelihood

    # ================================================================
    # ACTIVE LEARNING LOOP
    # ================================================================
    def active_learn(self, state):

        self.logger.info(f"Using {len(self.doc_ids)} documents, batch_size={self.embedding_batch_size}")

        lengthscale = self.gp_config.get('lengthscale')
        signal_noise = self.gp_config.get('signal_noise')
        observation_noise = self.gp_config.get('observation_noise')
        query_rel_label = self.gp_config.get('query_rel_label')
        k_final = int(self.gp_config.get('k_final'))

        warm_start_percent = float(self.gp_config.get('warm_start_percent', 0))

        self.al_config = self.config.get('active_learning', {})
        acq_func_name = self.al_config.get('acquisition_f')
        self.ard = self.opt_config.get('ard')

        for k in [
            "selected_doc_ids",
            "acquisition_scores",
            "acquisition_times",
            "acquisition_IO_times",
            "model_update_times",
            "neg_mll",
            "lengthscale",
            "signal_noise",
            "obs_noise"
        ]:
            state[k] = []

        X_obs = state["query_emb"].unsqueeze(0).to(self.device)
        y_obs = torch.tensor([query_rel_label], dtype=torch.float32).to(self.device)

        remaining_obs_post_ws = self.n_obs_iterations

        # =========================
        # WARM START
        # =========================
        if warm_start_percent > 0:
            top_k_psgs = state.get('top_k_psgs', [])
            if not top_k_psgs:
                raise ValueError("Warm start requested but 'top_k_psgs' not found in state")

            n_candidates = len(top_k_psgs)

            if warm_start_percent >= 100.0:
                n_warm = n_candidates
            else:
                n_warm = int(np.floor(n_candidates * (warm_start_percent / 100.0)))
                if n_warm <= 0:
                    n_warm = 1

            n_warm = min(n_warm, n_candidates)

            self.logger.info(
                f"Warm start enabled: percent={warm_start_percent}, "
                f"n_candidates={n_candidates}, n_warm={n_warm}"
            )

            docid_to_idx = {d_id: i for i, d_id in enumerate(self.doc_ids)}

            warm_added = 0
            for d_id in top_k_psgs[:n_warm]:
                idx = docid_to_idx.get(d_id, None)
                if idx is None:
                    self.logger.warning(
                        f"Warm start doc_id {d_id} not found in loaded doc_ids; skipping."
                    )
                    continue

                y_new = self.get_single_rel_judgment(state, d_id)

                X_new = torch.from_numpy(
                    self.index.reconstruct(idx)
                ).float().unsqueeze(0).to(self.device)

                X_obs = torch.cat([X_obs, X_new], dim=0)
                y_obs = torch.cat(
                    [y_obs, torch.tensor([y_new], dtype=torch.float32).to(self.device)],
                    dim=0
                )

                state["selected_doc_ids"].append(d_id)
                warm_added += 1

            if warm_added > 0:
                remaining_obs_post_ws = max(0, self.n_obs_iterations - warm_added)
                self.logger.info(
                    f"Warm start added {warm_added} labeled docs — "
                    f"reducing AL iterations to {remaining_obs_post_ws}"
                )


        for iteration in range(remaining_obs_post_ws):

            model, likelihood = self._build_and_maybe_refit_gp(
                state,
                X_obs,
                y_obs,
                lengthscale=lengthscale,
                signal_noise=signal_noise,
                observation_noise=observation_noise,
            )

            unobserved_indices = [
                i for i in range(len(self.doc_ids))
                if self.doc_ids[i] not in state["selected_doc_ids"]
            ]

            best_idx_in_unobs, acq_score, acq_gp_time, acq_io_time = \
                self.compute_acquisition_scores(
                    model,
                    unobserved_indices,
                    acq_func_name
                )

            selected_idx = unobserved_indices[best_idx_in_unobs]
            selected_doc_id = self.doc_ids[selected_idx]

            state["selected_doc_ids"].append(selected_doc_id)
            state["acquisition_scores"].append(acq_score)
            state["acquisition_times"].append(acq_gp_time)
            state["acquisition_IO_times"].append(acq_io_time)

            y_new = self.get_single_rel_judgment(state, selected_doc_id)

            X_new = torch.from_numpy(
                self.index.reconstruct(selected_idx)
            ).float().unsqueeze(0).to(self.device)

            X_obs = torch.cat([X_obs, X_new], dim=0)
            y_obs = torch.cat(
                [y_obs, torch.tensor([y_new], dtype=torch.float32).to(self.device)],
                dim=0
            )

        # FINAL MODEL
        model, likelihood = self._build_and_maybe_refit_gp(
            state,
            X_obs,
            y_obs,
            lengthscale=lengthscale,
            signal_noise=signal_noise,
            observation_noise=observation_noise,
        )

        # FINAL RANKING
        n_total = self.index.ntotal
        batch_size = self.embedding_batch_size
        n_batches = math.ceil(n_total / batch_size)

        posterior_scores = []
        final_gp_time = 0.0
        final_io_time = 0.0

        with torch.no_grad(), self.fast_ctx:
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_total)

                io_start = time.time()
                batch_embs = torch.from_numpy(
                    self.index.reconstruct_n(start, end - start)
                ).float().to(self.device)
                final_io_time += time.time() - io_start

                gp_start = time.time()
                pred_batch = model(batch_embs)

                if self.gp_task_type == "variational_binary":
                    probs = likelihood(pred_batch).mean
                    posterior_scores.extend(probs.cpu().tolist())

                elif self.gp_task_type == "binary_classification":
                    samples = pred_batch.sample(torch.Size((self.num_prob_samples,)))
                    samples = samples.exp()
                    probs = (samples / samples.sum(-2, keepdim=True)).mean(0)
                    posterior_scores.extend(probs[1].cpu().tolist())

                else:
                    posterior_scores.extend(pred_batch.mean.cpu().tolist())

                final_gp_time += time.time() - gp_start

        state["final_inf_time"] = final_gp_time
        state["final_IO_time"] = final_io_time

        sorted_indices = sorted(
            range(len(posterior_scores)),
            key=lambda i: posterior_scores[i],
            reverse=True
        )

        state["top_k_psgs"] = [self.doc_ids[i] for i in sorted_indices[:k_final]]

        if "query_emb" in state:
            state.pop("query_emb")


    def compute_acquisition_scores(self, model, unobserved_indices, acq_func_name):
        self.logger.debug(
            f"Computing acquisition scores using '{acq_func_name}' for {len(unobserved_indices)} unobserved documents"
        )

        n_unobs = len(unobserved_indices)
        batch_size = self.embedding_batch_size
        best_score = float('-inf')
        best_idx_in_unobs = -1
        total_io_time = 0.0
        total_gp_time = 0.0

        if acq_func_name == 'random':
            best_idx_in_unobs = torch.randint(0, n_unobs, (1,)).item()
            best_score = 0.0
            return best_idx_in_unobs, best_score, 0.0, 0.0

        with torch.no_grad(), self.fast_ctx:
            for i in range(0, n_unobs, batch_size):
                end = min(i + batch_size, n_unobs)
                batch_indices = unobserved_indices[i:end]

                io_start = time.time()
                batch_embs_np = np.array([self.index.reconstruct(idx) for idx in batch_indices])
                batch_embs = torch.from_numpy(batch_embs_np).float().to(self.device)
                total_io_time += time.time() - io_start

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

                total_gp_time += time.time() - gp_start

                if batch_max_score > best_score:
                    best_score = batch_max_score
                    best_idx_in_unobs = i + batch_max_idx

                del batch_embs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return best_idx_in_unobs, best_score, total_gp_time, total_io_time


    def _get_mu_sigma_for_acq(self, model, batch_embs):
        pred = model(batch_embs)

        if self.gp_task_type == "binary_classification":
            mu = pred.mean[1]
            sigma = pred.stddev[1]
            return mu, sigma, pred

        elif self.gp_task_type == "variational_binary":
            # latent GP — mean/std still exist in logit space
            mu = pred.mean
            sigma = pred.stddev
            return mu, sigma, pred

        else:
            mu = pred.mean
            sigma = pred.stddev
            return mu, sigma, pred


    def _ts_batch(self, model, batch_embs):
        pred = model(batch_embs)
        samples = pred.sample()

        if self.gp_task_type == "binary_classification":
            scores = samples[1]
        else:
            scores = samples

        max_score, max_idx = torch.max(scores, dim=0)
        return max_score.item(), max_idx.item()


    def _ucb_batch(self, model, batch_embs):
        beta = float(self.al_config['ucb_beta_const'])
        sqrt_beta = math.sqrt(beta)

        mu, sigma, _ = self._get_mu_sigma_for_acq(model, batch_embs)
        scores = mu + sqrt_beta * sigma

        max_score, max_idx = torch.max(scores, dim=0)
        return max_score.item(), max_idx.item()


    def _greedy_batch(self, model, batch_embs):
        mu, _, _ = self._get_mu_sigma_for_acq(model, batch_embs)
        max_score, max_idx = torch.max(mu, dim=0)
        return max_score.item(), max_idx.item()


    def _greedy_epsilon_batch(self, model, batch_embs):
        epsilon = self.config.get('active_learning', {}).get('epsilon')

        if torch.rand(1).item() > epsilon:
            return self._greedy_batch(model, batch_embs)
        else:
            batch_size = batch_embs.size(0)
            random_idx = torch.randint(0, batch_size, (1,)).item()
            return float('-inf'), random_idx


    def _greedy_epsilon_ts_batch(self, model, batch_embs):
        epsilon = self.config.get('active_learning', {}).get('epsilon')

        if torch.rand(1).item() > epsilon:
            return self._greedy_batch(model, batch_embs)
        else:
            return self._ts_batch(model, batch_embs)


    def _lse_straddle_batch(self, model, batch_embs):
        tau = float(self.al_config.get("lse_tau"))
        kappa = float(self.al_config.get("lse_kappa"))

        mu, sigma, _ = self._get_mu_sigma_for_acq(model, batch_embs)
        scores = -torch.abs(mu - tau) + kappa * sigma

        max_score, max_idx = torch.max(scores, dim=0)
        return max_score.item(), max_idx.item()


    def _lse_margin_batch(self, model, batch_embs):
        tau = float(self.al_config.get("lse_tau"))

        mu, sigma, _ = self._get_mu_sigma_for_acq(model, batch_embs)
        scores = -torch.abs(mu - tau) / (sigma + 1e-8)

        max_score, max_idx = torch.max(scores, dim=0)
        return max_score.item(), max_idx.item()


# ================================================================
# MODELS
# ================================================================
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
        self.covar_module.base_kernel.initialize(lengthscale=lengthscale)
        self.covar_module.initialize(outputscale=signal_noise)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DirichletExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, lengthscale, signal_noise, ard=False):
        super().__init__(train_x, train_y, likelihood)

        batch_shape = torch.Size((num_classes,))
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        if ard:
            d = train_x.size(-1)
            base_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=d)
        else:
            base_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape)

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=batch_shape)
        self.covar_module.base_kernel.initialize(lengthscale=lengthscale)
        self.covar_module.initialize(outputscale=signal_noise)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGPClassificationModel(ApproximateGP):
    def __init__(self, train_x, lengthscale, signal_noise, ard=False):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))

        variational_strategy = VariationalStrategy(
            self,
            train_x,
            variational_distribution,
            learn_inducing_locations=False,
        )

        super().__init__(variational_strategy)

        if ard:
            d = train_x.size(-1)
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=d)
        else:
            base_kernel = gpytorch.kernels.RBFKernel()

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        self.covar_module.base_kernel.initialize(lengthscale=lengthscale)
        self.covar_module.initialize(outputscale=signal_noise)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

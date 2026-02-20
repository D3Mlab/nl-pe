from abc import ABC
from contextlib import nullcontext
from nl_pe.utils.setup_logging import setup_logging
from nl_pe.utils.qrels import load_qrels_map, GTScorer
from nl_pe.llm.prompter import Prompter
from nl_pe.llm.ce import CEScorer #add sentence tranformer alternatives if needed
import os
import torch
import gpytorch
import pickle
import faiss
import numpy as np
import math
import time
import pandas as pd


# SCORER_CLASSES = {
#     'gt': GTScorer,
#     #"ce": CEScorer,
#     #"llm": Prompter,
# }

class BaseActiveLearner(ABC):

    def __init__(self, config, scorer):

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")
        self.n_obs_iterations = self.config.get('active_learning', {}).get('n_obs_iterations')

        # Set device
        tensor_ops_device = self.config.get('tensor_ops_device', 'cpu')
        self.device = torch.device('cuda' if tensor_ops_device == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        self.scorer = scorer
        self.score = scorer.score


class GPActiveLearner(BaseActiveLearner):

    def __init__(self, config, scorer):
        super().__init__(config, scorer)

        self.normalize_observations = bool(
            self.config.get('observation', {}).get('normalize_scores', False)
        ) 

        # Data config for index and batch size
        data_config = self.config.get('data', {})
        index_path = data_config.get('index_path')
        self.index = faiss.read_index(index_path)
        # Load all embeddings into CPU torch tensor once
        self.d_embs_cpu = torch.from_numpy(self.index.reconstruct_n(0, self.index.ntotal)).float().pin_memory()
        del self.index
        doc_ids_path = data_config.get('doc_ids_path')
        
        #read doc ids as strings
        doc_ids_path = data_config.get('doc_ids_path')
        raw_doc_ids = pickle.load(open(doc_ids_path, 'rb'))
        self.doc_ids = [str(d) for d in raw_doc_ids]   # <-- KEY FIX

        self.embedding_batch_size = data_config.get('embedding_batch_size', len(self.doc_ids))

        # GP config
        self.gp_config = self.config.get('gp', {})

        #learning config
        self.opt_config = self.config.get('optimization', {})

        fast_pred = self.gp_config.get("fast_pred", False)
        self.fast_ctx = gpytorch.settings.fast_pred_var() if fast_pred else nullcontext()
        if fast_pred:
            self.logger.info("Using fast_pred_var")

    def _get_gp_train_targets(self, y_obs):
        """
        Return training targets for GP model.

        If observation.normalize_scores is enabled, normalize using ALL
        observations collected so far (i.e., the full y_obs each time we build
        or rebuild the model).
        """
        y_train = y_obs.clone()

        if not self.normalize_observations:
            return y_train
        
        if y_train.numel() < 10:
            return y_train  #only normalize if over 10 observations (allows the query and reformulations to act as seed)

        mean = y_train.mean()
        std = y_train.std(unbiased=False)

        if std.item() < 1e-8:
            return y_train - mean

        return (y_train - mean) / (std + 1e-8)

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
        #optimization
        self.ard = self.opt_config.get('ard')
        #query reformulation
        use_query_reforms = str(self.gp_config.get('use_query_reformulations', False)).lower() in ("1", "true", "yes", "y")
        reform_query_rel_label = self.gp_config.get('reform_query_rel_label')

        #warm start percent: none or 0 to 100      
        warm_start_percent = float(self.gp_config.get('warm_start_percent', 0))

        # Active learning config
        self.al_config = self.config.get('active_learning', {})
        acq_func_name = self.al_config.get('acquisition_f') #low level af, eg UCB, greedy-epsilon, etc (could be used as a component in diversified acq like MMR, fantasy-UCB, etc)
        acq_strategy_name = self.al_config.get('acquisition_strategy', 'batch_af') #high-level acquisition strategy, eg 'batch_af','mmr_af', 'fantasy_af', etc which may use the low-level acquisition function as a component.
        acq_strategy = getattr(self, acq_strategy_name)
        k_acq = int(self.al_config.get("k_acq", 1))  # how many top-k candidates each acquisition call returns


        # Initialize lists
        state["selected_doc_ids"] = []
        state["observed_scores"] = []
        state["acquisition_scores"] = []
        state["neg_mll"] = []
        state["lengthscale"] = []
        state["signal_noise"] = []
        state["obs_noise"] = []
        state["model_update_times"] = []
        state["observation_times"] = []
    
        #read scorer cache
        prompt_name = self.config.get('templates', {}).get('pw_prompt', '')
        qid = state['qid']
        self.scorer.open_cache(qid,prompt_name=prompt_name)

        n_total = self.d_embs_cpu.shape[0]
        observed_mask_cpu = torch.zeros(n_total, dtype=torch.bool) #track which of the doc indicies have been observed

        
        # First observation: query_embedding and its label
        X_obs = state["query_emb"].unsqueeze(0).to(self.device)
        y_obs = torch.tensor([query_rel_label], dtype=torch.float32).to(self.device)
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
                ).to(self.device)
                X_obs = torch.cat([X_obs, reform_embs.to(self.device)], dim=0)
                y_obs = torch.cat([y_obs, reform_y], dim=0)
                self.logger.debug(
                    f"Added {n_reforms} query reformulation embeddings "
                    f"with label {reform_query_rel_label} to initial observations"
                )
            else:
                self.logger.warning(
                    "use_query_reformulations=True but no valid reformulation embeddings "
                    "or reform_query_rel_label is None; skipping reformulations."
                )

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
                warm_start_doc_ids = [str(d_id) for d_id in top_k_psgs[:n_warm]]

                self.logger.info(
                    f"Warm start enabled: percent={warm_start_percent}, "
                    f"n_candidates={n_candidates}, n_warm={n_warm}"
                )

                # Map doc_id -> index in doc_ids
                docid_to_idx = {d_id: i for i, d_id in enumerate(self.doc_ids)}

                warm_added = 0
                warm_doc_ids = []
                warm_doc_indices = []
                for d_id in warm_start_doc_ids:
                    idx = docid_to_idx.get(d_id, None)
                    if idx is None:
                        self.logger.warning(
                            f"Warm start doc_id {d_id} not found in loaded doc_ids; skipping."
                        )
                        continue
                    warm_doc_ids.append(d_id)
                    warm_doc_indices.append(idx)

                if warm_doc_ids:
                    warm_doc_ids = [str(i) for i in warm_doc_ids]

                    # Score warm-start docs in k_acq-sized batches (same pattern as AL loop)
                    warm_batch_size = max(k_acq, 1)
                    for start in range(0, len(warm_doc_ids), warm_batch_size):
                        end = min(start + warm_batch_size, len(warm_doc_ids))
                        batch_doc_ids = warm_doc_ids[start:end]
                        batch_indices = warm_doc_indices[start:end]

                        batch_labels = self.score(state, batch_doc_ids)

                        # Batch update observations
                        X_new = self.d_embs_cpu[batch_indices].to(self.device)
                        y_new_tensor = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
                        X_obs = torch.cat([X_obs, X_new], dim=0)
                        y_obs = torch.cat([y_obs, y_new_tensor], dim=0)

                        # Track selected docs / observed labels
                        state["selected_doc_ids"].extend(batch_doc_ids)
                        state["observed_scores"].extend([float(y_new) for y_new in batch_labels])

                        # Mark observations in mask
                        for idx in batch_indices:
                            observed_mask_cpu[idx] = True

                        warm_added += len(batch_doc_ids)

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

        # BO iterations (batch acquisitions)
        total_batches = math.ceil(remaining_obs_post_ws / max(k_acq, 1))
        for iteration in range(total_batches):
            self.logger.debug(
                f"Active learning iteration {iteration + 1}/{total_batches}"
            )

            model, likelihood = self._build_and_maybe_refit_gp(
                state,
                X_obs,
                y_obs,
                lengthscale=lengthscale,
                signal_noise=signal_noise,
                observation_noise=observation_noise,
            )

            # Get acquisition scores for all docs except observed
            top_idxs, top_scores, = acq_strategy(
                state,
                model,
                observed_mask_cpu,
                acq_func_name,
                k_acq=k_acq,
            )

            del model, likelihood

            # Batch update for top-k acquisitions
            remaining_slots = remaining_obs_post_ws - (iteration * k_acq) #correct for last batch which may have fewer than k_acq slots
            batch_size = min(len(top_idxs), remaining_slots)
            selected_indices = top_idxs[:batch_size]

            selected_doc_ids = [str(self.doc_ids[idx]) for idx in selected_indices]
            self.logger.debug(
                "Selected %d documents for acquisition (head=%s)",
                len(selected_doc_ids),
                selected_doc_ids[:5],
            )

            # Record acquisition metadata (store all scores returned for this step)
            state["selected_doc_ids"].extend(selected_doc_ids)
            state["acquisition_scores"].extend(top_scores[:batch_size])

            for idx in selected_indices:
                observed_mask_cpu[idx] = True

            # Get labels for selected docs (batch)
            y_new_batch = self.score(state, selected_doc_ids)
            state["observed_scores"].extend([float(y_new) for y_new in y_new_batch])

            # Update observations with batch
            X_new = self.d_embs_cpu[selected_indices].to(self.device)
            y_new_tensor = torch.tensor(y_new_batch, dtype=torch.float32).to(self.device)
            X_obs = torch.cat([X_obs, X_new], dim=0)
            y_obs = torch.cat([y_obs, y_new_tensor], dim=0)
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
                batch_embs = self.d_embs_cpu[start:end].to(self.device)
                io_time = time.time() - io_start
                final_io_time += io_time
                gp_start = time.time()
                pred_batch = model(batch_embs)
                gp_time = time.time() - gp_start
                final_gp_time += gp_time
                posterior_means.extend(pred_batch.mean.tolist())
                del batch_embs, pred_batch
                #if torch.cuda.is_available():
                #    torch.cuda.empty_cache()

        state["final_inf_time"] = final_gp_time
        state["final_IO_time"] = final_io_time
        self.logger.debug("Creating final ranked list from posterior means")
        sorted_indices = sorted(range(len(posterior_means)), key=lambda i: posterior_means[i], reverse=True)
        state["top_k_psgs"] = [self.doc_ids[i] for i in sorted_indices[:k_final]]

        # pop embeddings
        if "query_emb" in state:
            state.pop("query_emb")
        if "query_reformation_embeddings" in state:
            state.pop("query_reformation_embeddings")

        #write scorer cache
        self.scorer.write_cache()

        self.logger.debug(f"Final ranked list created with top 5 docs: {state['top_k_psgs'][:5]}")

        del model, likelihood
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
    
    def batch_af(self, state, model, observed_mask_cpu, acq_func_name, k_acq=1, sorted=True):

        self.logger.debug(f"Computing acquisition scores using '{acq_func_name}'")
        n_total = self.d_embs_cpu.shape[0]
        batch_size = self.embedding_batch_size

        if sorted:
            #incumbent top-k scores and indicies
            inc_scores = None
            inc_indices = None
        else:
            dense_scores = torch.full((n_total,), float("-inf"), dtype=torch.float32)

        total_io_time = 0.0
        total_gp_time = 0.0
        total_sort_time = 0.0

        observed_mask_gpu = observed_mask_cpu.to(self.device)

        # For other methods, batch process
        with torch.no_grad(), self.fast_ctx:
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                # IO time: retrieving embeddings from pre-loaded tensor
                io_start = time.time()
                batch_embs = self.d_embs_cpu[start:end].to(self.device,non_blocking=True)
                batch_obs = observed_mask_gpu[start:end]
                io_time = time.time() - io_start
                total_io_time += io_time

                # GP time: model predictions
                gp_start = time.time()
                if acq_func_name == "ts":
                    scores = self._ts_batch(model, batch_embs)
                elif acq_func_name == "ucb_const_beta":
                    scores = self._ucb_batch(model, batch_embs)
                elif acq_func_name == "greedy":
                    scores = self._greedy_batch(model, batch_embs)
                elif acq_func_name == 'greedy_epsilon':
                    scores = self._greedy_epsilon_batch(model, batch_embs)
                elif acq_func_name == "lse_straddle":
                    scores = self._lse_straddle_batch(model, batch_embs)
                elif acq_func_name == "lse_margin":
                    scores = self._lse_margin_batch(model, batch_embs)

                else:
                    raise ValueError(f"Unknown acquisition function: {acq_func_name}")

                total_gp_time += time.time() - gp_start

                #apply mask to observed cands
                scores[batch_obs] = float("-inf")

                ####################################################
                # UNSORTED PATH (FAST — NO TOPK)
                ####################################################
                if not sorted:
                    dense_scores[start:end] = scores.detach().cpu()
                    continue

                ####################################################
                # ORIGINAL SORTED PATH
                ####################################################

                sort_start = time.time()
                k_here = min(k_acq, scores.numel())
                batch_top_scores, batch_top_local = torch.topk(scores, k=k_here, largest=True)
                batch_top_global = batch_top_local + start

                batch_top_scores = batch_top_scores.detach()
                batch_top_global = batch_top_global.detach()

                if inc_scores is None:
                    inc_scores = batch_top_scores
                    inc_indices = batch_top_global
                else:
                    merged_scores = torch.cat([inc_scores, batch_top_scores], dim=0)
                    merged_indices = torch.cat([inc_indices, batch_top_global], dim=0)

                    k_merge = min(k_acq, merged_scores.numel())
                    inc_scores, pos = torch.topk(merged_scores, k=k_merge, largest=True)
                    inc_indices = merged_indices[pos]
                total_sort_time += time.time() - sort_start


        if "inner_acquisition_times" not in state:
            state["inner_acquisition_times"] = []
        if "inner_acquisition_IO_times" not in state:
            state["inner_acquisition_IO_times"] = []
        if "inner_acquisition_sort_times" not in state:
            state["inner_acquisition_sort_times"] = []

        #record times - 'inner' since could be an outer af like fantasy-ucb
        state["inner_acquisition_times"].append(round(total_gp_time,3))
        state["inner_acquisition_IO_times"].append(round(total_io_time,3))
        state["inner_acquisition_sort_times"].append(round(total_sort_time,3))

        ###########################################################
        # RETURN
        ###########################################################

        if sorted:
            return inc_indices.cpu().tolist(), inc_scores.cpu().tolist()
        else:
            # return ids in natural order + aligned scores
            return list(range(n_total)), dense_scores.tolist()
        


    def fantasy_af(self, state, model, observed_mask_cpu, acq_func_name, k_acq=1):
        pass
        #DEEPCOPY the model to pass into batch_af

    def mmr_af(self, state,model, observed_mask_cpu, acq_func_name, k_acq=1): 
        """
        Exact greedy MMR.

        Algorithm:
            1. Compute base acquisition scores ONCE for all candidates.
            2. Select the first candidate purely via AF (no diversity penalty).
            3. For each newly selected candidate:
                - run ONE batched similarity sweep against all docs
                - update max_sim
                - recompute MMR scores
                - select next

        IMPORTANT:
            We perform exactly ONE KNN-style similarity pass per selected document.
        """

        outer_start = time.time()
        mmr_lambda = float(self.al_config.get("mmr_lambda"))

        n_total = self.d_embs_cpu.shape[0]
        batch_size = n_total #if start to OOM, decrease batch size to smaller or self.embedding_batch_size

        ##########################################################
        # 1. Compute base AF scores ONCE (already batched internally)
        ##########################################################

        _, af_scores = self.batch_af(
            state,
            model,
            observed_mask_cpu,
            acq_func_name,
            #k_acq=n_total,  #k_acq ignored if unsorted
            sorted=False,   # get unsorted scores to align with doc indicies for masking and later similarity computations
        )
        af_scores = torch.tensor(af_scores, dtype=torch.float32)

        ##########################################################
        # 2. Initialize MMR state
        ##########################################################

        selected = []
        selected_scores = []

        available_mask = ~observed_mask_cpu.clone() #flips observed_mask to get available mask, which we will update as we select candidates

        # tracks max similarity to ANY selected doc
        max_sim = torch.zeros(n_total, dtype=torch.float32)

        mmr_knn_time = 0.0

        num_to_select = min(k_acq, int(available_mask.sum().item()))

        ##########################################################
        # 3. FIRST PICK — PURE AF (no similarity yet)
        ##########################################################

        masked_af = af_scores.masked_fill(~available_mask, float("-inf"))
        next_idx = int(torch.argmax(masked_af).item())

        selected.append(next_idx)
        selected_scores.append(float(af_scores[next_idx]))
        available_mask[next_idx] = False

        ##########################################################
        # 4. Remaining picks (k_acq - 1)
        #    ONE similarity sweep per selected candidate
        ##########################################################

        for _ in range(num_to_select - 1):

            knn_start = time.time()

            # vector of newly selected document
            picked_vec = self.d_embs_cpu[next_idx:next_idx+1].to(self.device) #torch tensor shape (1, d) for matmul

            with torch.no_grad():

                # batched matrix mult against ALL documents
                for start in range(0, n_total, batch_size):
                    end = min(start + batch_size, n_total)

                    chunk = self.d_embs_cpu[start:end].to(self.device) #[B, d]

                    # cosine == dot if embeddings are normalized
                    sims = torch.matmul(chunk, picked_vec.T).squeeze(1) #[B]

                    # update running maximum similarity
                    max_sim[start:end] = torch.maximum(
                        max_sim[start:end], 
                        sims.cpu()
                    ) #elemntwise max to update max_sim with the new sims if they are higher for the slice of candidates in this batch

                    del chunk, sims

            mmr_knn_time += time.time() - knn_start

            ######################################################
            # Compute MMR scores and select next
            ######################################################

            mmr_scores = mmr_lambda * af_scores - (1 - mmr_lambda) * max_sim
            mmr_scores[~available_mask] = float("-inf")

            next_idx = int(torch.argmax(mmr_scores).item())

            selected.append(next_idx)
            selected_scores.append(float(mmr_scores[next_idx]))
            available_mask[next_idx] = False

        ##########################################################
        # 5. Record timing
        ##########################################################

        if "outer_acquisition_times" not in state:
            state["outer_acquisition_times"] = []
        if "mmr_knn_times" not in state:
            state["mmr_knn_times"] = []

        state["outer_acquisition_times"].append(
            round(time.time() - outer_start, 3)
        )

        state["mmr_knn_times"].append(
            round(mmr_knn_time, 3)
        )

        return selected, selected_scores


    def _ts_batch(self, model, batch_embs):
        pred = model(batch_embs)
        scores = pred.sample()
        return scores

    def _ucb_batch(self, model, batch_embs):
        if 'ucb_beta_const' not in self.al_config:
            raise KeyError("UCB acquisition requires 'ucb_beta_const' in config['active_learning']")
        beta = float(self.al_config['ucb_beta_const'])
        pred = model(batch_embs)
        scores = pred.mean + math.sqrt(beta) * pred.stddev
        return scores

    def _greedy_batch(self, model, batch_embs):
        pred = model(batch_embs)
        scores = pred.mean
        return scores

    def _greedy_epsilon_batch(self, model, batch_embs):
        epsilon = self.config.get('active_learning', {}).get('epsilon')

        # Exploit: greedy scores
        if torch.rand(1).item() > epsilon:
            return self._greedy_batch(model, batch_embs)

        # Explore: random scores for entire batch
        batch_size = batch_embs.size(0)
        return torch.rand(
            batch_size,
            device=batch_embs.device,
        )

    def _lse_straddle_batch(self, model, batch_embs):
        tau = float(self.al_config.get("lse_tau"))
        kappa = float(self.al_config.get("lse_kappa"))

        pred = model(batch_embs)
        mu = pred.mean
        sigma = pred.stddev

        scores = -torch.abs(mu - tau) + kappa * sigma
        return scores

    def _lse_margin_batch(self, model, batch_embs):
        tau = float(self.al_config.get("lse_tau"))

        pred = model(batch_embs)
        mu = pred.mean
        sigma = pred.stddev

        scores = -torch.abs(mu - tau) / (sigma + 1e-8)
        return scores

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

        y_train = self._get_gp_train_targets(y_obs)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=observation_noise)
        likelihood = likelihood.to(self.device)

        model = ExactGPModel(
            X_obs,
            y_train,
            likelihood,
            lengthscale,
            signal_noise,
            ard = self.ard
        ).to(self.device)

        # log initial GP hyperparameters
        with torch.no_grad():
            ls = model.covar_module.base_kernel.lengthscale.detach().cpu()
            if ls.numel() == 1:
                ls_log = float(ls.item())
            else:
                ls_log = ls.view(-1).tolist()

            sig_noise_log = float(model.covar_module.outputscale.item())
            obs_noise_log = float(likelihood.noise.item())

        # self.logger.debug(
        #     "Initialized GP hypers | ard=%s | lengthscale=%s | signal_noise=%.6f | obs_noise=%.6f",
        #     self.ard,
        #     ls_log,
        #     sig_noise_log,
        #     obs_noise_log,
        # )

        self._maybe_refit_gp(state, model, likelihood, X_obs, y_train)

        elapsed = time.time() - start
        state["model_update_times"].append(round(elapsed,3))

        model.eval()
        likelihood.eval()

        return model, likelihood

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











from pyexpat import model
import yaml
import json
import argparse
from dotenv import load_dotenv
import os
from pathlib import Path
import time
import pandas as pd
from nl_pe.utils.setup_logging import setup_logging
from nl_pe.utils.gps import * 
from nl_pe.embedding import EMBEDDER_CLASSES
from nl_pe import search_agent
import pickle
import torch
import faiss
import gpytorch
from gpytorch.constraints import GreaterThan
import math

class ExperimentManager():

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.load_config()
        self.setup_logger()
        self.config['exp_dir'] = self.exp_dir

    def index_corpus(self):
        self.logger.info("Starting corpus indexing...")

        self.embedding_config = self.config.get('embedding', {})
        self.data_config = self.config.get('data', {})

        embedder_class = EMBEDDER_CLASSES[self.embedding_config.get('class')]
        self.embedder = embedder_class(self.config)

        #get index method
        index_method_name = self.embedding_config.get('index_method', '')
        self.index_method = getattr(self.embedder, index_method_name)

        start_time = time.time()

        self.index_method(
            texts_csv_path = self.data_config.get('d_text_csv', ''),
            index_path = self.data_config.get('index_path', ''),
            inference_batch_size = self.embedding_config.get('inference_batch_size', None),
            prompt = self.embedding_config.get('doc_prompt', '')
        )

        end_time = time.time()
        embedding_time = end_time - start_time

        embedding_details_path = os.path.join(self.exp_dir, "detailed_results.json")
        with open(embedding_details_path, 'w') as f:
            json.dump({'embedding_time': embedding_time}, f)

    def ir_exp(self):
        self.logger.info("Starting IR experiment...")

        self.data_config = self.config.get('data', {})

        agent_class = search_agent.AGENT_CLASSES[self.config.get('agent', {}).get('agent_class')]
        self.agent = agent_class(self.config)

        self.results_dir = Path(self.exp_dir) / 'per_query_results'
        self.results_dir.mkdir(exist_ok=True)

        queries_path = self.data_config.get('q_text_csv', '')
        qs_df = pd.read_csv(queries_path, header=0)
        qids = qs_df.iloc[:, 0].tolist()
        queries = qs_df.iloc[:, 1].tolist()

        for qid, query in zip(qids, queries):
            try:
                self.logger.info(f"Ranking query {qid}: {query}")

                state = {
                "query": query,
                'qid': str(qid),
                'terminate': False
                }

                result = self.agent.act(state)

                if result['top_k_psgs']:
                    self.logger.info('Rank successful')
                    self.write_query_result(qid, result)
                else:
                    self.logger.error(f'Failed to rank query {qid} -- empty result[\'top_k_psgs\']')

            except Exception as e:
                self.logger.error(f'Failed to rank or write results for query {qid}: {str(e)}')

    def gp_inf(self):
        self.logger.info("Starting GP Inference speed experiment...")
        from nl_pe.gp_tests.inference import GPInference
        gp = GPInference(self.config)
        gp.run_inference()


    def tune_gp(self):

        self.logger.info(f"Starting gp tuning in {self.exp_dir}")
        self.data_config = self.config.get('data', {})

        # --------------------------------------------------
        # Device setup
        # --------------------------------------------------
        device_cfg = self.config.get("device", "cpu")
        device = torch.device("cuda" if device_cfg == "gpu" else "cpu")
        self.logger.info(f"Using device: {device}")

        #Move to data construction
        # --------------------------------------------------
        # Load queries
        # --------------------------------------------------
        # queries_csv_path = self.data_config.get("queries_csv_path")
        # qdf = pd.read_csv(queries_csv_path)
        # self.qids = qdf.iloc[:, 0].tolist()
        # self.logger.info(f"Loaded {len(self.qids)} training queries")

        # --------------------------------------------------
        # Construct training data
        # --------------------------------------------------
        f_get_train_set = getattr(self, self.config.get("pretraining").get("data_constr_func"))

        X, Y = f_get_train_set()          # X: QxKxD, Y: QxK
        X = X.to(device)
        Y = Y.to(device)

        self.Q, self.K, self.emb_dim = X.shape

        self.logger.info(
            "Constructed training set via %s (Q=%d, K=%d, D=%d)",
            f_get_train_set,
            self.Q,
            self.K,
            self.emb_dim,
        )

        # --------------------------------------------------
        # Shared GP components
        # --------------------------------------------------
        f_get_mean = getattr(self, self.config.get("pretraining").get("mean_constr_func"))
        f_get_kernel = getattr(self, self.config.get("pretraining").get("kernel_constr_func"))
        f_get_likelihood = getattr(self, self.config.get("pretraining").get("likelihood_constr_func"))
        f_make_single_model = getattr(self, self.config.get("pretraining").get("single_model_constr_func"))

        shared_mean = f_get_mean().to(device)
        shared_kernel = f_get_kernel().to(device)
        shared_likelihood = f_get_likelihood().to(device)

        # --------------------------------------------------
        # Build per-query GP submodels
        # --------------------------------------------------
        self.logger.info(f"Creating {self.Q} GP submodels (K={self.K}, D={self.emb_dim})")

        models = []
        likelihoods = []

        for q in range(self.Q):
            train_x_q = X[q]   # K x D
            train_y_q = Y[q]   # K

            model_q = f_make_single_model(
                train_x_q,
                train_y_q,
                shared_likelihood,
                shared_kernel,
                shared_mean,
            ).to(device)

            models.append(model_q)
            likelihoods.append(shared_likelihood)

        # --------------------------------------------------
        # Wrap models + likelihoods
        # --------------------------------------------------
        model = gpytorch.models.IndependentModelList(*models).to(device)
        likelihood = gpytorch.likelihoods.LikelihoodList(*likelihoods).to(device)

        # --------------------------------------------------
        # Optimize
        # --------------------------------------------------
        lr = self.config.get('optimization').get('lr')
        train_iters = self.config.get('optimization').get('train_iters')
        
        #whether to optimizer observation and signal noise 
        opt_noise = bool(self.config.get('optimization').get('opt_noise'))
        opt_sig_noise = bool(self.config.get('optimization').get('opt_sig_noise'))

        model.train()
        likelihood.train()


       


    #helper methods for training
    def _get_zero_mean(self):
        return gpytorch.means.ZeroMean()
    
    def _get_rbf_kernel(self):
        self.ard = self.config.get('optimization').get('ard')
        if self.ard:
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.emb_dim)
        else:
            base_kernel = gpytorch.kernels.RBFKernel()
        return gpytorch.kernels.ScaleKernel(base_kernel)
    
    def _get_gaussian_likelihood(self):
        return gpytorch.likelihoods.GaussianLikelihood()
    
    def _make_SharedKernelAndLikelihoodGPModel(self, train_x, train_y, likelihood, covar_module, mean_module):
        return SharedKernelAndLikelihoodGPModel(train_x, train_y, likelihood, covar_module, mean_module)
    
    #for testing
    def _dummy_data_constr(self):
        # two queries, 25 points each, 2D inputs
        train_x1_1d = torch.linspace(0, 0.95, 25) + 0.05 * torch.rand(25)
        train_x2_1d = torch.linspace(0, 0.95, 25) + 0.05 * torch.rand(25)

        # make inputs 2D: (x, x^2) just as a simple example
        train_x1 = torch.stack([train_x1_1d, train_x1_1d ** 2], dim=-1)  # 25 x 2
        train_x2 = torch.stack([train_x2_1d, train_x2_1d ** 2], dim=-1)  # 25 x 2

        train_y1 = torch.sin(train_x1_1d * (2 * math.pi)) + 0.2 * torch.randn_like(train_x1_1d)
        train_y2 = torch.cos(train_x2_1d * (2 * math.pi)) + 0.2 * torch.randn_like(train_x2_1d)

        # stack into Q x K x D and Q x K
        X = torch.stack([train_x1, train_x2], dim=0)  # 2 x 25 x 2
        Y = torch.stack([train_y1, train_y2], dim=0)  # 2 x 25

        return X, Y




    def tune_gp_all_queries(self):
        #needs to be corrected -- it uses only on GP model for all queries, with y's sampled in batches for training from different queries
        """
        Tune shared GP hyperparameters over all training queries.

        - x_vals: fixed document embeddings from FAISS (N x D)
        - For each query q: y_q over same docs from qrels
        - Objective: minimize average neg marginal log-likelihood over queries
        """
        import random

        self.logger.info("Starting GP tuning experiment over all queries...")

        # -------------------------
        # 1. Load doc_ids and FAISS index -> x_vals
        # -------------------------
        self.data_config = self.config.get('data', {})
        doc_ids_path = self.data_config.get('doc_ids_path')

        with open(doc_ids_path, 'rb') as f:
            doc_ids = pickle.load(f)

        index_path = self.data_config.get('index_path')
        index = faiss.read_index(index_path)
        n = index.ntotal
        d = index.d
        self.logger.info(f"FAISS index loaded with {n} vectors of dim {d}")

        xb_np = index.reconstruct_n(0, n)  # shape (n, d), float32
        x_vals = torch.from_numpy(xb_np)   # (N, D)

        self.logger.info(
            "Built x_vals tensor from FAISS index: shape=%s, device=%s, dtype=%s",
            x_vals.shape, x_vals.device, x_vals.dtype
        )

        # -------------------------
        # 2. Load all query IDs
        # -------------------------
        queries_csv_path = self.data_config.get("queries_csv_path")
        qdf = pd.read_csv(queries_csv_path)
        all_qids = qdf.iloc[:, 0].tolist()
        self.logger.info(f"Loaded {len(all_qids)} training queries")

        # -------------------------
        # 3. Load qrels and precompute y for each query
        # -------------------------
        qrels_path = self.data_config.get("qrels_path")

        qrels = {}
        with open(qrels_path, "r") as f:
            for line in f:
                qid, _, docid, rel = line.strip().split()
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = int(rel)

        self.logger.info(f"Loaded qrels for {len(qrels)} queries from {qrels_path}")

        # Precompute y vectors per query: y_q[i] = rel(q, doc_ids[i]) or 0
        qid_to_y = {}
        num_docs = len(doc_ids)
        for qid in all_qids:
            rel_map = qrels.get(qid, {})
            y = torch.zeros(num_docs, dtype=torch.float32)
            for i, doc_id in enumerate(doc_ids):
                if str(doc_id) in rel_map:
                    y[i] = rel_map[str(doc_id)]
            qid_to_y[qid] = y

        self.logger.info("Precomputed y vectors for all queries")

        # -------------------------
        # 4. Set up GP model with shared hyperparameters
        # -------------------------
        # Use the first query's y just to initialize the model; we'll train on all y's.
        first_qid = all_qids[0]
        y_init = qid_to_y[first_qid]

        gp_log_path = os.path.join(self.config["exp_dir"], "gp_all_queries_tuning_log.csv")
        with open(gp_log_path, "w") as f:
            f.write("step,avg_neg_mll,lengthscale,signal_var,noise_var,lr,num_queries_in_batch\n")

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=GreaterThan(0.1)
        )
        model = ExactGPModel(x_vals, y_init, likelihood)

        model.train()
        likelihood.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # -------------------------
        # 5. Multi-query training loop (mini-batches over queries)
        # -------------------------
        num_steps = 2000          # upper bound, early stopping will cut this
        query_batch_size = 16    # number of queries per step

        best_loss = float("inf")
        no_improve_steps = 0
        patience_early_stop = 10
        min_delta = 1e-4
        min_lr = 1e-5

        for step in range(num_steps):
            optimizer.zero_grad()

            # Sample a mini-batch of queries
            batch_qids = random.sample(
                all_qids,
                k=min(query_batch_size, len(all_qids))
            )

            # Forward pass is the same for all queries (same x_vals)
            output = model(x_vals)

            total_neg_mll = 0.0
            for qid in batch_qids:
                y_q = qid_to_y[qid]
                total_neg_mll = total_neg_mll - mll(output, y_q)

            avg_neg_mll = total_neg_mll / len(batch_qids)
            avg_neg_mll.backward()

            optimizer.step()
            scheduler.step(avg_neg_mll.item())

            curr_lr = optimizer.param_groups[0]["lr"]
            loss_val = avg_neg_mll.item()

            # Early stopping logic on average neg_mll
            if loss_val < best_loss - min_delta:
                best_loss = loss_val
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            # Extract shared hyperparameters
            lengthscale = model.covar_module.base_kernel.lengthscale.item()
            signal_var = model.covar_module.outputscale.item()
            noise_var = likelihood.noise.item()

            # Log to CSV
            with open(gp_log_path, "a") as f:
                f.write(
                    f"{step},{loss_val},{lengthscale},"
                    f"{signal_var},{noise_var},{curr_lr},{len(batch_qids)}\n"
                )

            # Log to experiment logger
            self.logger.info(
                f"Step {step}: avg_neg_mll={loss_val:.4f}, "
                f"lengthscale={lengthscale:.4f}, signal_var={signal_var:.4f}, "
                f"noise_var={noise_var:.4f}, lr={curr_lr:.6f}, "
                f"batch_queries={len(batch_qids)}"
            )

            if no_improve_steps >= patience_early_stop and curr_lr <= min_lr:
                self.logger.info(
                    f"Early stopping at step {step}: "
                    f"avg_neg_mll={loss_val:.4f}, best={best_loss:.4f}, lr={curr_lr:.6f}"
                )
                break


    def tune_gp_first_query(self):
        #rough function to test gp tuning on a single query, remove later
        self.logger.info("Starting GP tuning experiment on a single query...")

        self.data_config = self.config.get('data', {})
        doc_ids_path = self.data_config.get('doc_ids_path')
        #unpickle to get doc ids, its in the same order as the faiss index
        with open(doc_ids_path, 'rb') as f:
            doc_ids = pickle.load(f)

        #path to faiss index
        index_path = self.data_config.get('index_path')

        #build x_vals tensor (use torch) for each vector in the index, in order
        index = faiss.read_index(index_path)
        n = index.ntotal
        d = index.d
        self.logger.info(f"FAISS index loaded with {n} vectors of dim {d}")
        xb_np = index.reconstruct_n(0, n)  # shape (n, d), float32
        #device = torch.device(self.tensor_ops_device)
        x_vals = torch.from_numpy(xb_np)#.to(device)

        self.logger.info(
            "Built x_vals tensor from FAISS index: shape=%s, device=%s, dtype=%s",
            x_vals.shape, x_vals.device, x_vals.dtype
        )

        queries_csv_path = self.data_config.get("queries_csv_path")
        qdf = pd.read_csv(queries_csv_path)
        curr_qid = qdf.iloc[1, 0]
        self.logger.info(f"Using first query ID: {curr_qid}")

        qrels_path = self.data_config.get("qrels_path")

        qrels = {}
        with open(qrels_path, "r") as f:
            for line in f:
                qid, _, docid, rel = line.strip().split()
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = int(rel)

        rel_map = qrels.get(curr_qid, {})
        y_vals = torch.zeros(len(doc_ids), dtype=torch.float32)

        for i, doc_id in enumerate(doc_ids):
            if str(doc_id) in rel_map:
                y_vals[i] = rel_map[str(doc_id)]

        #Ok, now we're going to tune the gp hyperparms with an rbf kernel: lengthscale, signal variance, noise variance
        #what to store in gp_single_query_tuning_log.csv in the experiment directory? headers as step, neg_mll, lengthscale, signal_var, noise_var
        #output csv name: gp_single_query_tuning_log.csv
                # ================================
        # Train a GP on (x_vals, y_vals)
        # ================================
        gp_log_path = os.path.join(self.config["exp_dir"], "gp_single_query_tuning_log.csv")

        # Write header
        with open(gp_log_path, "w") as f:
            f.write("step,neg_mll,lengthscale,signal_var,noise_var\n")

        # GP model
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(x_vals, y_vals, likelihood)

        model.train()
        likelihood.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

        # Marginal log-likelihood object
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        num_steps = 200  # upper bound, we'll early-stop anyway

        best_loss = float("inf")
        no_improve_steps = 0
        patience_early_stop = 10   # how many steps with no improvement before stopping
        min_delta = 1e-4           # minimum improvement to count
        min_lr = 1e-5              # don't early stop until LR has shrunk enough

        for step in range(num_steps):
            optimizer.zero_grad()

            output = model(x_vals)
            neg_mll = -mll(output, y_vals)   # this is the "loss" we minimize
            neg_mll.backward()

            optimizer.step()
            scheduler.step(neg_mll.item())

            curr_lr = optimizer.param_groups[0]["lr"]
            loss_val = neg_mll.item()

            # Early stopping logic
            if loss_val < best_loss - min_delta:
                best_loss = loss_val
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            # Extract hyperparameters
            lengthscale = model.covar_module.base_kernel.lengthscale.item()
            signal_var = model.covar_module.outputscale.item()
            noise_var = likelihood.noise.item()

            # Log to CSV
            with open(gp_log_path, "a") as f:
                f.write(f"{step},{loss_val},{lengthscale},{signal_var},{noise_var},{curr_lr}\n")

            # Log to experiment logger
            self.logger.info(
                f"Step {step}: neg_mll={loss_val:.4f}, "
                f"lengthscale={lengthscale:.4f}, signal_var={signal_var:.4f}, "
                f"noise_var={noise_var:.4f}, lr={curr_lr:.6f}"
            )

            if no_improve_steps >= patience_early_stop and curr_lr <= min_lr:
                self.logger.info(
                    f"Early stopping at step {step}: "
                    f"neg_mll={loss_val:.4f}, best={best_loss:.4f}, lr={curr_lr:.6f}"
                )
                break


    def write_query_result(self, qid, result):
        """
        Write two files: 
        1) TREC run file : trec_results_raw.txt (may have duplicates from LLM reranking)
        2) JSON: detailed_results.json
        """
        #clean result
        result.pop('query_emb', None)

        query_result_dir = self.results_dir / f"{qid}"
        query_result_dir.mkdir(exist_ok=True)
        detailed_results_path = query_result_dir / "detailed_results.json"
        trec_file_path = query_result_dir / "trec_results_raw.txt"

        with open(detailed_results_path, 'w') as file:
            json.dump(result, file, indent=4)

        trec_results = []
        top_k_psgs = result.get('top_k_psgs', [])
        for p_rank, pid in enumerate(top_k_psgs):
            score = len(top_k_psgs) - p_rank
            trec_results.append(f"{qid} Q0 {pid} {p_rank + 1} {score} run")

        with open(trec_file_path, "w") as trec_file:
            trec_file.write("\n".join(trec_results))


    def load_config(self):
        config_path = os.path.join(self.exp_dir, "config.yaml")
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

    def setup_logger(self):
        self.logger = setup_logging(self.__class__.__name__, self.config, output_file=os.path.join(self.exp_dir, "experiment.log"))

    def run_experiment(self, exp_type):
        # Call the method dynamically
        if not hasattr(self, exp_type):
            raise ValueError(f"Experiment type '{exp_type}' is not defined in ExperimentManager.")
        method = getattr(self, exp_type)
        method()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments in the specified directory.")
    parser.add_argument("-c", "--exp-dir", type=str, required=True, help="Path to the experiment directory containing config.yaml")
    parser.add_argument("-e", "--exp-type", type=str, required=True, help="Name of the experiment method to run (e.g., index_corpus)")
    args = parser.parse_args()

    load_dotenv()

    config_path = os.path.join(args.exp_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"No config.yaml found in {args.exp_dir}. Skipping experiment.")
    else:
        manager = ExperimentManager(args.exp_dir)
        manager.run_experiment(args.exp_type)

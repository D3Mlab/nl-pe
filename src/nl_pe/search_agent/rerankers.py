from nl_pe.utils.setup_logging import setup_logging
import os
import time

class TextReranker():
    def __init__(self, config, scorer):

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")

        self.scorer = scorer
        self.score = scorer.score

        #batch/sw size
        self.k_acq = int(self.config.get('active_learning', {}).get("k_acq"))

        # max number of docs to score (like AL budget)
        self.n_obs_iterations = int(
            self.config.get('active_learning', {}).get('n_obs_iterations', 0)
        )


class TopKPWReranker(TextReranker):

    def rerank(self, state):
        #open scorer cache
        prompt_name = self.config.get('templates', {}).get('pw_prompt', '')
        qid = state['qid']
        self.scorer.open_cache(qid,prompt_name=prompt_name)

        doc_ids = [str(d) for d in state['top_k_psgs']]

        # Respect observation budget
        if self.n_obs_iterations and self.n_obs_iterations > 0:
            doc_ids = doc_ids[:self.n_obs_iterations]

        n_total = len(doc_ids)
        
        state['observation_times'] = []
        all_scores = []

        # Process in batches preserving original order
        for start in range(0, n_total, self.k_acq):
            end = min(start + self.k_acq, n_total)
            batch_doc_ids = doc_ids[start:end]

            # EXACT same pattern as active learner
            batch_scores = self.score(state, batch_doc_ids)

            # ensure float
            all_scores.extend([float(s) for s in batch_scores])

        # Pair doc_ids with scores
        doc_score_pairs = list(zip(doc_ids, all_scores))

        # Sort descending by score
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Unzip
        sorted_doc_ids = [d for d, _ in doc_score_pairs]
        sorted_scores = [float(s) for _, s in doc_score_pairs]

        # Update state
        # Only reorder the scored subset
        original = [str(d) for d in state['top_k_psgs']]

        # Append untouched tail (if any)
        remaining = original[len(sorted_doc_ids):]

        state['top_k_psgs'] = sorted_doc_ids + remaining
        state['top_k_rel_scores'] = sorted_scores

        # write scorer cache
        self.scorer.write_cache()




class LWReranker(TextReranker):

    def rerank(self, state):
        #open scorer cache
        prompt_name = self.config.get('templates', {}).get('lw_prompt', '')
        qid = state['qid']
        self.scorer.open_cache(qid,prompt_name=prompt_name)

        #TODO: tail prepend dense order

        #implement a sliding window reranker. Sliding window size is self.k_acq, and the overlap is always half the windwow (round if needed)

        #start at the bottom, and get those k_acq doc_ids, and pass them to scorer.lw_rerank(state, batch_doc_ids)
        #

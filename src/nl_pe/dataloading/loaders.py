from abc import ABC, abstractmethod
from nl_pe.utils.setup_logging import setup_logging
import json
import os
from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
import csv

class BaseLoader(ABC):

    def __init__(self, config):
        self.config = config
        self.data_config = config.get('data', {})
        self.logger = setup_logging(self.__class__.__name__, self.config)
        self.logger.info(f"Initializing...")
        self.topics = None
        self.index = None 

    def ensure_sorted_run_file(self, run_path):
        with open(run_path, 'r') as f:
            lines = f.readlines()

        sorted_lines = sorted(lines, key=lambda x: (x.split()[0], int(x.split()[3])))
        if lines == sorted_lines:
            self.logger.debug(f"Using correctly sorted run file: {run_path}")
            return run_path

        sorted_run_path = run_path.replace('.txt', '_sorted.txt')
        with open(sorted_run_path, 'w') as f:
            f.writelines(sorted_lines)
        
        self.logger.warning(f"Run file was not sorted by qid and rank. A sorted version has been created at {sorted_run_path}")
        return sorted_run_path

    @abstractmethod
    def get_qs_from_run(self, run_path):
        # get list of all queries used in a run file
        # return: [{'qid':_, 'text': _},...]
        pass

    @abstractmethod
    def get_psgs_from_run(self, run_path, qid):
        # for qid, get list of all passages used in a run file
        # return: [{'pid':_, 'text': _},...]
        pass
    

class PLLoader(BaseLoader):

    def __init__(self, config):
        super().__init__(config)
        self.queries_path = os.path.join(self.data_config.get('queries_path', ''))
        self.passages_path = os.path.join(self.data_config.get('passages_path', ''))

    def get_qs_from_run(self, run_path):
        query_list = []
        seen_qids = set()
        with open(self.queries_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            queries = {row['QueryID']: row['Query'] for row in reader}
        with open(run_path, 'r') as f:
            for line in f:
                qid, _, _, _, _, _ = line.strip().split()
                if qid not in seen_qids:
                    query_list.append({'qid': qid, 'text': queries[qid]})
                    seen_qids.add(qid)
        return query_list

    
    def get_psgs_from_run(self, run_path, qid):
        qid = str(qid)
        with open(self.passages_path, 'r') as f:
            passages = json.load(f)
        passage_list = []
        with open(run_path, 'r') as f:
            for line in f:
                line_qid, _, pid, _, _, _ = line.strip().split()
                if line_qid == qid:
                    passage_list.append({'pid': pid, 'text': passages[pid]})
                elif line_qid > qid:
                    break
        return passage_list

class PyseriniLoader(BaseLoader):       

    def __init__(self, config):
        super().__init__(config)
        self.topics = self.data_config.get('topics')
        self.index= self.data_config.get('index')
        self.searcher = LuceneSearcher.from_prebuilt_index(self.index)

    def get_qs_from_run(self, run_path):
        topics = get_topics(self.topics)
        query_list = []
        seen_qids = set()
        with open(run_path, 'r') as f:
            for line in f:
                qid, _, _, _, _, _ = line.strip().split()
                qid = int(qid)
                if qid not in seen_qids:
                    query_list.append({'qid': qid, 'text': topics[qid]['title']})
                    seen_qids.add(qid)
        return query_list

    def get_psgs_from_run(self, run_path, qid):
        qid = str(qid)
        passage_list = []
        with open(run_path, 'r') as f:
            for line in f:
                line_qid, _, pid, _, _, _ = line.strip().split()
                if line_qid == qid:
                    doc = json.loads(self.searcher.doc(pid).raw())
                    if 'contents' in doc:
                        text = doc['contents']
                    elif 'text' in doc:
                        text = doc['text']
                        if 'title' in doc:
                            text = f"{doc['title']} {text}"
                    else:
                        self.logger.error(f"Document {pid} does not have 'contents' or 'text' fields.")
                        return None
                    passage_list.append({'pid': pid, 'text': text})
                elif line_qid > qid:
                    break
        return passage_list

class TestLoader(BaseLoader):

    def __init__(self, config):
        super().__init__(config)
        self.dataset = "test"
        self.queries_path = os.path.join(self.data_config.get('queries_path', ''))
        self.passages_path = os.path.join(self.data_config.get('passages_path', ''))

    def get_qs_from_run(self, run_path):
        with open(self.queries_path, 'r') as f:
            queries = json.load(f)
        query_list = []
        seen_qids = set()
        with open(run_path, 'r') as f:
            for line in f:
                qid, _, _, _, _, _ = line.strip().split()
                if qid not in seen_qids:
                    query_list.append({'qid': qid, 'text': queries[qid]['text']})
                    seen_qids.add(qid)
        return query_list

    def get_psgs_from_run(self, run_path, qid):
        qid = str(qid)
        with open(self.passages_path, 'r') as f:
            passages = json.load(f)
        passage_list = []
        with open(run_path, 'r') as f:
            for line in f:
                line_qid, _, pid, _, _, _ = line.strip().split()
                if line_qid == qid:
                    passage_list.append({'pid': pid, 'text': passages[pid]['text']})
                elif line_qid > qid:
                    break
        return passage_list

if __name__ == "__main__":
    #tests for PLLoader
    config = {
        'data': {
            'queries_path': 'data/pl_test_set_2/queries.tsv',
            'passages_path': 'data/pl_test_set_2/passages.json'
        }
    }
    run_path = 'data/pl_test_set_2/run.txt'
    loader = PLLoader(config)
    query_list = loader.get_qs_from_run(run_path)
    
    # Print the first five queries
    for query in query_list[:5]:
        print(query)
    
    # Print the length of the query list
    print(f"Total number of queries: {len(query_list)}")
    
    # Test get_psgs_from_run for the first query
    if query_list:
        first_qid = query_list[0]['qid']
        passage_list = loader.get_psgs_from_run(run_path, first_qid)
        
        # Print the first five passages
        for passage in passage_list[:5]:
            print(passage)
        
        # Print the length of the passage list
        print(f"Total number of passages for query {first_qid}: {len(passage_list)}")




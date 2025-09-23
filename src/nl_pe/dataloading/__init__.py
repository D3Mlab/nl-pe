from llm_passage_ranking.dataloading.loaders import TestLoader
from llm_passage_ranking.dataloading.loaders import PyseriniLoader

LOADER_CLASSES = {
    "TestLoader": TestLoader,
    "PyseriniLoader": PyseriniLoader
}

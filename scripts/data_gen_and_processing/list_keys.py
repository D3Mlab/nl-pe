import json
from collections.abc import Mapping, Sequence

PATH = "trials/ir/nfcorpus/gp/tuned/miniLM/ts/per_query_results/PLAIN-2/detailed_results.json"

def walk(obj, prefix=""):
    """
    Recursively walk containers and print only lowest-level containers
    (i.e., containers whose elements are all non-containers).
    """
    is_container = isinstance(obj, (Mapping, Sequence)) and not isinstance(obj, (str, bytes))

    if not is_container:
        return

    # dict
    if isinstance(obj, Mapping):
        if obj and all(
            not isinstance(v, (Mapping, Sequence)) or isinstance(v, (str, bytes))
            for v in obj.values()
        ):
            print(f"{prefix or '<root>'}: {len(obj)}")
        else:
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))

    # list / tuple
    else:
        if obj and all(
            not isinstance(v, (Mapping, Sequence)) or isinstance(v, (str, bytes))
            for v in obj
        ):
            print(f"{prefix or '<root>'}: {len(obj)}")
        else:
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]")


with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

walk(data)
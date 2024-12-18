from collections import defaultdict
from typing import Hashable, Iterable, Tuple


def get_minimal_collections(
    *collections: Iterable[Hashable],
) -> Tuple[Iterable[Hashable], ...]:
    minimal_collections = defaultdict(lambda: None)  # order-preserving

    for i, collection in enumerate(collections):
        for j, collection_other in enumerate(collections):
            if i == j:
                continue
            if collection >= collection_other:
                break
        else:
            minimal_collections[collection]

    return tuple(minimal_collections)


# TODO: rename file (no sets are even used)
# collections?

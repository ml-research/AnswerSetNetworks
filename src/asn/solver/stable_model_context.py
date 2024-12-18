from collections import namedtuple

StableModelContext = namedtuple(
    "StableModelContext", ["atoms", "is_SM", "npp_choices", "inverse_sink_ids"]
)

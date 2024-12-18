# stlib
import argparse
import json
import os
import warnings
from pathlib import Path
from time import perf_counter, strftime

warnings.filterwarnings("ignore")

# clingo
import clingo

# PyTorch
import torch

# GroundSLASH
from ground_slash.grounding import Grounder
from ground_slash.program import Program

# ASN
from asn.asn import ASN
from asn.solver import SolvingContext

# ----- parse arguments -----

parser = argparse.ArgumentParser()

parser.add_argument("--title", "--t", type=str, default="n_queens")
parser.add_argument("--num-queens", "--n", type=int, default=4)
parser.add_argument("--num-phases", "--p", type=int, default=1)
parser.add_argument("--device", "--d", type=str, default="cpu")
parser.add_argument("--num_runs", "--r", nargs="+", type=int, default=[2, 10])
parser.add_argument("--log-path", "--lpath", type=str, default="./logs/")

args = parser.parse_args()

# check number of queens
assert (
    args.num_queens >= 0
), f"Number of queens must be greater of equal to zero, but was {args.num_queens}."
# check number of phases
assert (
    args.num_phases > 0
), f"Number of phases must be greater than zero, but was {args.num_phases}."
# check device
assert torch.device(args.device), f"{args.device} is no valid device."
# check number of runs
assert len(args.num_runs) in (
    1,
    2,
), "Number of runs is expected to be one or two integers."
assert (
    args.num_runs[-1] > 0
), f"Number of runs must be greater than zero, but was {args.num_runs[-1]}."
if len(args.num_runs) > 1:
    assert (
        args.num_runs[0] >= 0
    ), f"Number of warmup runs must be greater or equal to zero, but was {args.num_runs[0]}."
# create log path if it does not exist yet
if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)

print("----- n-queens experiment -----")
print(f"title: {args.title}")
print(f"num-queens: {args.num_queens}")
print(f"num-phases: {args.num_phases}")
print(f"device: {args.device}")
print(f"num_runs: {args.num_runs}")
print(f"path: {args.log_path}")
print("-----")

# ----- set up experiment -----

print("initializing experiment...")

# create program string
prog_str = ""

# initialize rows
for n in range(args.num_queens):
    prog_str += f"n({n}).\n"

# choose a column for each row
prog_str += (
    "1={" + ";".join([f"q(X,{n})" for n in range(args.num_queens)]) + "} :- n(X).\n"
)

# no column overlap
prog_str += ":- q(X1,Y), q(X2,Y), X1<X2.\n"
# no diagonal overlap
prog_str += ":- q(X1,Y1), q(X2,Y2), n(N), X2=X1+N, Y2=Y1+N, N>0.\n"
prog_str += ":- q(X1,Y1), q(X2,Y2), n(N), X2=X1+N, Y1=Y2+N, N>0."

# ground program
grounder = Grounder(Program.from_string(prog_str))
grnd_prog = grounder.ground()

# initialize experiment log
exp_log = {
    "title": args.title,
    "num_queens": args.num_queens,
    "num-phases": args.num_phases,
    "device": args.device,
    "num_runs": args.num_runs,
    "date": strftime("%Y%m%d-%H%M%S"),
    "prog": prog_str.split("\n"),
    "clingo": {
        "t_init": None,
        "t_solving_readout": None,
        "t_total": None,
        "solutions": set(),
    },
    "asn": {
        "t_init": None,
        "t_batching": None,
        "t_solving": None,
        "t_readout": None,
        "t_total": None,
        "solutions": set(),
    },
    "valid": None,  # whether or not clingo & asn produce same solutions
    "complete": False,  # whether or not the experiment fully finished
}

# open file for storing experiment log
log_path = Path(args.log_path, f"{args.title}.json")

# encoder to convert sets to lists for JSON serialization
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

with log_path.open("w") as f:
    json.dump(exp_log, f, indent=4, cls=SetEncoder)

# ----- clingo -----

print("clingo...")

t_init_cum = 0.0
t_solving_readout_cum = 0.0
t_total_cum = 0.0

for r in range(sum(args.num_runs)):
    print(f"run {r+1}...")

    t_start = perf_counter()

    ctl = clingo.Control(message_limit=0)
    # instruct to return all models
    ctl.configuration.solve.models = 0
    ctl.add("prog", [], str(grnd_prog))
    # ground (necessary although already grounded)
    ctl.ground([("prog", [])])

    t_init = perf_counter()

    # solve
    ctl.solve(
        on_model=lambda m: exp_log["clingo"]["solutions"].add(
            frozenset(str(m).split(" "))
        )
    )

    t_end = perf_counter()

    # accumulate timings if warmup has passed
    if r >= sum(args.num_runs) - args.num_runs[-1]:
        t_init_cum += t_init - t_start
        t_solving_readout_cum += t_end - t_init
        t_total_cum += t_end - t_start

exp_log["clingo"]["t_init"] = t_init_cum / args.num_runs[-1]
exp_log["clingo"]["t_solving_readout"] = t_solving_readout_cum / args.num_runs[-1]
exp_log["clingo"]["t_total"] = t_total_cum / args.num_runs[-1]

print("average time:", exp_log["clingo"]["t_total"])

with log_path.open("w") as f:
    json.dump(exp_log, f, indent=4, cls=SetEncoder)

# ----- ASN -----

print("asn...")

t_init_cum = 0.0
t_batching_cum = 0.0
t_solving_cum = 0.0
t_readout_cum = 0.0
t_total_cum = 0.0

for r in range(sum(args.num_runs)):
    print(f"run {r+1}...")

    t_batching_phase_cum = 0.0
    t_solving_phase_cum = 0.0

    t_start = perf_counter()

    # initialize solver
    asn = ASN(grnd_prog, False, grounder=grounder, num_phases=args.num_phases)

    # initialize solving context
    solving_ctx = SolvingContext()

    t_init = perf_counter()

    for phase in range(args.num_phases):
        t_phase_start = perf_counter()

        # prepare graph block
        graph_block = asn.prepare_block(
            rg=asn.rg,  # pass pre-computed reasoning graph (avoids copying)
            phase=phase,
            device=args.device,
        )

        t_batching = perf_counter()

        # solve graph block
        graph_block = asn.solve(graph_block)

        t_solving = perf_counter()

        solving_ctx.update_SMs(graph_block)

        t_batching_phase_cum += t_batching - t_phase_start
        t_solving_phase_cum += t_solving - t_batching

    # get labels for SMs
    exp_log["asn"]["solutions"] = set(
        frozenset(
            label
            for label, atom in zip(asn.rg.node_dict["atom"]["label"], atoms)
            if torch.isclose(atom, torch.ones_like(atom))
        )
        for atoms in solving_ctx.sm_ctx.atoms[solving_ctx.sm_ctx.is_SM[0].squeeze(-1)]
    )

    t_end = perf_counter()

    # accumulate timings if warmup has passed
    if r >= sum(args.num_runs) - args.num_runs[-1]:
        t_init_cum += t_init - t_start
        t_batching_cum += t_batching_phase_cum
        t_solving_cum += t_solving_phase_cum
        t_readout_cum += t_end - t_solving
        t_total_cum += t_end - t_start

exp_log["asn"]["t_init"] = t_init_cum / args.num_runs[-1]
exp_log["asn"]["t_batching"] = t_batching_cum / args.num_runs[-1]
exp_log["asn"]["t_solving"] = t_solving_cum / args.num_runs[-1]
exp_log["asn"]["t_readout"] = t_readout_cum / args.num_runs[-1]
exp_log["asn"]["t_total"] = t_total_cum / args.num_runs[-1]

# TODO: across all graph blocks
#exp_log["asn"]["num_nodes"] = {
#    node_type: solving_ctx.node_dict[node_type]["num_nodes"]
#    for node_type in ("atom", "disj", "conj", "count", "sum", "min", "max")
#}
#exp_log["asn"]["num_edges"] = {
#    "\t".join(edge_type): edge_attrs["edge_index"].shape[1]
#    for edge_type, edge_attrs in solving_ctx.edge_dict.items()
#}
print("average time:", exp_log["asn"]["t_total"])

with log_path.open("w") as f:
    json.dump(exp_log, f, indent=4, cls=SetEncoder)

# ----- compare -----

print("comparing stable models...", end="")

exp_log["valid"] = exp_log["clingo"]["solutions"] == exp_log["asn"]["solutions"]
exp_log["complete"] = True

print(exp_log["valid"])

with log_path.open("w") as f:
    json.dump(exp_log, f, indent=4, cls=SetEncoder)

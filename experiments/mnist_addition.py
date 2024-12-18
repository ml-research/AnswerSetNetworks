# stdlib
import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from time import perf_counter, strftime

warnings.filterwarnings("ignore")

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf

# GroundSLASH
from ground_slash.grounding import Grounder
from ground_slash.program import Program
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# ASN
from asn.asn import ASN
from asn.models.alexnet import AlexNet
from asn.solver import SolvingContext
from asn.data.datasets.mnist_addition import MNISTAddition

# ----- parse arguments -----

parser = argparse.ArgumentParser()

parser.add_argument("--title", "--t", type=str, default="mnist_addition")
parser.add_argument("--num-digits", "--n", type=int, default=2)
parser.add_argument("--classes", "--c", nargs="+", type=int, default=list(range(10)))
parser.add_argument("--learning-rate", "--lr", type=float, default=0.005)
parser.add_argument("--batch-size", "--bs", type=int, default=100)
parser.add_argument("--eval-batch-size", "--eval-bs", type=int, default=None)
parser.add_argument("--num-phases", "--p", type=int, default=1)
parser.add_argument("--num-epochs", "--e", type=int, default=100)
parser.add_argument("--seed", "--s", type=int, default=None)
parser.add_argument("--device", "--d", type=str, default="cpu")
parser.add_argument("--num-runs", "--r", nargs="+", type=int, default=[2, 10])
parser.add_argument("--log-path", "--lpath", type=str, default="./logs/")
parser.add_argument("--data-path", "--dpath", type=str, default="../data/")

args = parser.parse_args()

# check number of digits
assert (
    args.num_digits >= 2
), f"Number of digits must be greater of equal to two, but was {args.num_digits}."
# check classes
assert len(set(args.classes)) == len(args.classes), "Duplicate classes."
assert all(c >= 0 and c < 10 for c in args.classes), "Invalid classes."
# check learning rate
assert (
    args.learning_rate > 0.0
), f"Learning rate must be positive, but was {args.learning_rate}."
if args.eval_batch_size is None:
    args.eval_batch_size = args.batch_size
# check batch size
assert (
    args.batch_size > 0
), f"Batch size must greater than zero, but was {args.batch_size}."
# check eval batch size
assert (
    args.eval_batch_size > 0
), f"Evaluation batch size must greater than zero, but was {args.eval_batch_size}."
# check number of phases
assert (
    args.num_phases > 0
), f"Number of phases must be greater than zero, but was {args.num_phases}."
# check number of epochs
assert (
    args.num_epochs >= 0
), f"Number of epochs must be greater or equal to zero, but was {args.num_epochs}."
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

print("----- mnist-addition experiment -----")
print(f"title: {args.title}")
print(f"num-digits: {args.num_digits}")
print(f"classes: {args.classes}")
print(f"learning-rate: {args.learning_rate}")
print(f"batch-size: {args.batch_size}")
print(f"eval-batch-size: {args.eval_batch_size}")
print(f"num-phases: {args.num_phases}")
print(f"num-epochs: {args.num_epochs}")
print(f"seed: {args.seed}")
print(f"device: {args.device}")
print(f"num_runs: {args.num_runs}")
print(f"log-path: {args.log_path}")
print(f"data-path: {args.data_path}")
print("-----")

# ----- set up experiment -----

print("initializing experiment...")

# create program string
prog_str = ""

# initialize images
for n in range(args.num_digits):
    prog_str += f"img(i{n+1}).\n"

# NPPs
prog_str += f"#npp(digit(X), {args.classes}) :- img(X).\n"

# addition
prog_str += (
    "addition("
    # images
    + ",".join([f"i{n+1}" for n in range(args.num_digits)])
    + ","
    # sum of digits
    + "+".join([f"N{n+1}" for n in range(args.num_digits)])
    + ") :- "
    # individual digits
    + ", ".join([f"digit(i{n+1},N{n+1})" for n in range(args.num_digits)])
    # + ", "
    ## order of images
    # + ", ".join([f"X{n}<X{n+1}" for n in range(args.num_digits - 1)])
    + "."
)
# TODO: commutativity

# ground program
grounder = Grounder(Program.from_string(prog_str))
grnd_prog = grounder.ground()

# MNIST addition dataset
mnist_add = MNISTAddition(
    n=args.num_digits,
    root=args.data_path,
    train=True,
    transform=tf.Compose([tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
    download=True,
    digits=args.classes,
    seed=args.seed,
)
# original MNIST dataset
mnist_train = mnist_add.mnist
mnist_test = MNIST(
    root=args.data_path,
    train=False,
    transform=tf.Compose([tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
    download=True,
)

# data loader for single MNIST digits
mnist_train_loader = DataLoader(
    mnist_train, batch_size=args.eval_batch_size, shuffle=True
)
mnist_test_loader = DataLoader(
    mnist_test, batch_size=args.eval_batch_size, shuffle=True
)
# data loader for MNIST addition
mnist_addition_loader = DataLoader(mnist_add, batch_size=args.batch_size, shuffle=True)


# evaluation routine
def eval_loader(model: nn.Module, loader: DataLoader):
    n_correct = 0
    n_total = 0

    for x, y in loader:
        x = x.to(args.device)
        y = y.to(args.device)

        with torch.no_grad():
            y_pred = torch.argmax(model(x), dim=-1)
            n_correct += (y_pred == y).sum().cpu().tolist()
            n_total += len(y)

    return n_correct, n_total, float(n_correct) / n_total


# initialize experiment log
exp_log = {
    "title": args.title,
    "num_digits": args.num_digits,
    "classes": args.classes,
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "eval_batch_size": args.eval_batch_size,
    "num_phases": args.num_phases,
    "num_epochs": args.num_epochs,
    "seed": args.seed,
    "device": args.device,
    "num_runs": args.num_runs,
    "date": strftime("%Y%m%d-%H%M%S"),
    "prog": prog_str.split("\n"),
    "runs": [],
    "complete": False,
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

# -----

print("asn...")

for r in range(sum(args.num_runs)):
    print(f"run {r+1}...")

    # log for this run
    run_log = {
        "t_init": None,
        "epochs": [],
    }

    # create NPP model for digits
    model = AlexNet(len(args.classes))
    model.to(args.device)

    # evaluate model before training
    train_n_correct, train_n_total, train_acc = eval_loader(model, mnist_train_loader)
    test_n_correct, test_n_total, test_acc = eval_loader(model, mnist_test_loader)
    run_log["epochs"].append(
        {
            "t_npp_forward": None,
            "t_encode_queries": None,
            "t_batching": None,
            "t_solver": None,
            "t_grad": None,
            "t_update": None,
            "t_total": None,
            "loss": None,
            "train_correct": train_n_correct,
            "train_total": train_n_total,
            "train_accuracy": train_acc,
            "test_correct": test_n_correct,
            "test_total": test_n_total,
            "test_accuracy": test_acc,
            "num_nodes": {},
            "num_edges": {},
        }
    )

    t_start = perf_counter()

    # initialize solver
    asn = ASN.from_string(prog_str, num_phases=args.num_phases)

    # provide models and optimizers for NPPs
    # NOTE: only track optimizer for first digit since they share the same network (do not want multiple updates)
    asn.configure_NPPs(
        {
            npp_rule: {
                "model": model,
                "optimizer": optim.Adam(model.parameters(), lr=args.learning_rate)
                if not i
                else None,
            }
            for i, npp_rule in enumerate(asn.rg.npp_edges)
        }
    )

    t_init = perf_counter()
    run_log["t_init"] = t_init - t_start

    for e in range(args.num_epochs):
        print(f"\tepoch {e+1}/{args.num_epochs}...", end="")

        t_epoch_start = perf_counter()

        # running loss for epoch
        total_loss = torch.tensor(0.0, device=args.device)

        # cummulative timings
        t_npp_forward_cum = 0.0
        t_encode_queries_cum = 0.0
        t_batching_cum = 0.0
        t_solving_cum = 0.0
        t_SLASH_grad_cum = 0.0
        t_update_cum = 0.0

        num_nodes_cum = defaultdict(int)
        num_edges_cum = defaultdict(int)

        # for each batch
        for x, y in mnist_addition_loader:

            t_batch_start = perf_counter()


            # NPP forward pass
            npp_ctx_dict = asn.npp_forward(
                npp_data={
                    npp_rule: (x_i.to(args.device),)
                    for i, (npp_rule, x_i) in enumerate(zip(asn.rg.npp_edges, x))
                },
            )

            t_npp_forward = perf_counter()

            # encode queries in reasoning graph
            # NOTE: we can reuse this reasoning graph across all sequential blocks
            queries = mnist_add.to_queries(y)
            rg = asn.encode_queries(queries)

            t_encode_queries = perf_counter()

            # initialize solving context
            solving_ctx = SolvingContext(
                len(queries),
                npp_ctx_dict,
            )

            for phase in range(args.num_phases):
                t_phase_start = perf_counter()

                # prepare graph block
                graph_block = asn.prepare_block(
                    queries=queries,
                    rg=rg,  # pass pre-computed reasoning graph
                    phase=phase,
                    device=args.device,
                )

                t_batching = perf_counter()

                # solve graph block
                graph_block = asn.solve(graph_block)

                t_solving = perf_counter()

                # update stable models
                solving_ctx.update_SMs(graph_block)

                t_batching_cum += t_batching - t_phase_start
                t_solving_cum += t_solving - t_batching

                # accumulate nodes and edges
                for node_type in ("atom", "disj", "conj", "count", "sum", "min", "max"):
                    num_nodes_cum[node_type] += graph_block.node_dict[node_type][
                        "num_nodes"
                    ]
                for edge_type, edge_attrs in graph_block.edge_dict.items():
                    num_edges_cum["\t".join(edge_type)] += edge_attrs[
                        "edge_index"
                    ].shape[1]

            # compute loss and gradients
            loss = solving_ctx.npp_loss

            t_SLASH_grad = perf_counter()

            # zero gradients
            asn.zero_grad()

            # backward pass
            (-loss).backward()

            # update NPPs
            asn.step()

            t_update = perf_counter()

            # add loss to running loss
            total_loss += loss.detach()

            # accumulate timings
            t_npp_forward_cum += t_npp_forward - t_batch_start
            t_encode_queries_cum += t_encode_queries - t_npp_forward
            t_SLASH_grad_cum += t_SLASH_grad - t_solving
            t_update_cum += t_update - t_SLASH_grad

        t_epoch_end = perf_counter()

        # evaluate
        train_n_correct, train_n_total, train_acc = eval_loader(
            model, mnist_train_loader
        )
        test_n_correct, test_n_total, test_acc = eval_loader(model, mnist_test_loader)
        print("\t", t_epoch_end - t_epoch_start, total_loss, train_acc, test_acc)

        # store epoch statistics
        run_log["epochs"].append(
            {
                "t_npp_forward": t_npp_forward_cum,
                "t_encode_queries": t_encode_queries_cum,
                "t_batching": t_batching_cum,
                "t_solver": t_solving_cum,
                "t_SLASH_grad": t_SLASH_grad_cum,
                "t_update": t_update_cum,
                "t_total": t_epoch_end - t_epoch_start,
                "loss": -loss.detach().clone().cpu().tolist(),
                "train_correct": train_n_correct,
                "train_total": train_n_total,
                "train_accuracy": train_acc,
                "test_correct": test_n_correct,
                "test_total": test_n_total,
                "test_accuracy": test_acc,
                "num_nodes": {
                    node_type: float(node_count) / len(mnist_addition_loader)
                    for node_type, node_count in num_nodes_cum.items()
                },
                "num_edges": {
                    edge_type: float(edge_count) / len(mnist_addition_loader)
                    for edge_type, edge_count in num_edges_cum.items()
                },
            }
        )

    # accumulate timings if warmup has passed
    if r >= sum(args.num_runs) - args.num_runs[-1]:
        exp_log["runs"].append(run_log)

        # export statistics
        with log_path.open("w") as f:
            json.dump(exp_log, f, indent=4, cls=SetEncoder)

exp_log["complete"] = True

with log_path.open("w") as f:
    json.dump(exp_log, f, indent=4, cls=SetEncoder)

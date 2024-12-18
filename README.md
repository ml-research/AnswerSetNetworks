# Answer Set Networks:  Casting Answer Set Programming into Deep Learning
Arseny Skryagin, Daniel Ochs, Philipp Deibert, Simon Kohaut, Devendra Singh Dhami , Kristian Kersting

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/ml-research/AnswerSetNetworks/blob/main/imgs/logo_dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/ml-research/AnswerSetNetworks/blob/main/imgs/logo_light.png?raw=true">
  <img alt="Shows a black logo in light color mode and a white one in dark color mode." src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
</picture>
</p>

# Abstract
Although Answer Set Programming (ASP) allows constraining neural-symbolic (NeSy) systems, its employment is hindered by the prohibitive costs of computing stable models and the CPU-bound nature of state-of-the-art solvers.
To this end, we propose Answer Set Networks (ASN), a NeSy solver.
Based on Graph Neural Networks (GNN), ASNs are a scalable approach to ASP-based Deep Probabilistic Logic Programming (DPPL).
Specifically, we show how to translate ASPs into ASNs and demonstrate how ASNs can efficiently solve the encoded problem by leveraging GPU's batching and parallelization capabilities. 
Our experimental evaluations demonstrate that ASNs outperform state-of-the-art CPU-bound NeSy systems on multiple tasks.
Simultaneously, we make the following two contributions based on the strengths of ASNs.
Namely, we are the first to show the finetuning of Large Language Models (LLM) with DPPLs, employing ASNs to guide the training with logic.
Further, we show the "constitutional navigation" of drones, i.e., encoding public aviation laws in an ASN for routing Unmanned Aerial Vehicles in uncertain environments.




# Installation

### Environment
First you need a PyTorch environment. You can either use our prebuilt docker container available on the docker hub (see [hansiwusti/asn:1.0](https://hub.docker.com/r/hansiwusti/asn)) or create an environment yourself. For this we provided a `Dockerfile` and the `pyproject.toml`. Note that we found the environment with PyTorch==2.3.0 and PyTorch Geometric==2.5.3 to work well, but you may need to select a PyTorch version which fits to your own GPU/Cuda environment. 

### Cloning the repo and ASP Grrounder
After setting up your environment you need to install ASN and a grounder. We use the GroundSlash grounder from https://github.com/pdeibert/GroundSLASH for ASN.

Start by cloning the ASN and Grounder repositories
```
git clone git@github.com:pdeibert/AnswerSetNetworks.git
cd AnswerSetNetworks/
git clone git@github.com:pdeibert/GroundSLASH.git
```

Then install all python modules using:
```
python -m pip install --upgrade pip
python -m pip install -e . 
python -m pip install ./GroundSLASH
```
This will also install PyTorch and other requirements if not installed yet. 

### API keys and working dir
To set the working dir for all experiments you need to create a `.env` file:
```
#Environment file to specify workspace dir and API Keys
WORKSPACE_DIR="{smth like /workspaces/ASN_dev/AnswerSetNetworks}"

#Huggingface caches if you dont want to use the default cache
HF_HOME={your hf path here}
HUGGINGFACE_HUB_CACHE={your hf path here}

#weights and biases
WANDB_PROJECT=answer-set-networks
WANDB_API_KEY={insert your key here}
```

### LLMs in ASN
if you want to use ASN to train LLMs you have to install additional packages (Huggingface transformers, wandb, ...). In your project root run:
```
python -m pip install .[transformer_libs]
```


# Run ASN
We put together a folder containing all experiment scripts in '/experiments/scripts'. 
To start ASN for MNIST addition with two images you can run: 
```
. experiments/scripts/mnist_add.sh
```
This script will provide you with a good starting point to explore all python args. Also you can check out the other scripts in the folder to get an idea of how to start ASN for other experiments. The script will call the mnist_addition.py which exemplifies how to connect your PyTorch models to NPP objects, create a dataloader with Constraint as your labels and calls the ASN forward and backward pass.

FROM nvcr.io/nvidia/pytorch:24.06-py3
WORKDIR "/asn"
COPY . .
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN echo "Upgrade pip" 
RUN python -m pip install -U pip
RUN echo "Install torch==2.3.0 & dependencies..."
RUN python -m pip install torch==2.3.0 \ 
                          torchvision==0.18.0 \
                          torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN echo "Install torch_geometric==2.5.3 & dependencies..."
RUN python -m pip install pyg_lib \
                          torch_scatter \
                          torch_sparse \
                          torch_cluster \ 
                          torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
RUN python -m pip install torch_geometric==2.5.3
RUN echo "Install ASN grounder"
WORKDIR "/asn/GroundSLASH/"
RUN python -m pip install -e .
WORKDIR "/asn"
RUN echo "Install Graphviz development libraries"
RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev \
    pkg-config
RUN echo "Install ASN"
RUN python -m pip install -e .
RUN echo "All installations completed successfully!"
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html && \
    pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html && \
    pip install -U "ray[tune]"==2.0.0 && \
    pip install -U hyperopt==0.2.7 && \
    pip install matplotlib==3.5.3 && \
    pip install tensorboard==2.10.0 && \
    pip install markupsafe==2.0.1 && \
    pip install numba==0.56.0

ENV PYTHONPATH "${PYTHONPATH}:/workspace"
WORKDIR /workspace


CMD ["bash"]
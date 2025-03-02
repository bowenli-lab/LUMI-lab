FROM dptechnology/unicore:0.0.1-pytorch1.11.0-cuda11.3

RUN conda install -y -c conda-forge rdkit==2021.09.5 && conda clean -ya

RUN pip install transformers==4.41.2 wandb==0.16.6 toml==0.10.2 accelerate==0.31.0 datasets==2.19.2

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    conda clean -ya

WORKDIR /app
COPY ./serverless-infer /app
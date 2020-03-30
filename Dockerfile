FROM nvidia/cuda:10.1-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    nano \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /consistency
WORKDIR /consistency

# Pull the repo
RUN cd /consistency \
  && git init \
  && git remote add origin https://github.com/alexsax/midlevel-reps.git \
  && git pull origin visualpriors

# Make thos files viewable
RUN mkdir /consistency/scripts
RUN echo "default_job_name, 0, mount" > /consistency/scripts/jobinfo.txt
RUN chmod -R 770 /consistency


# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /consistency
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya

# CUDA 10.1-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    "pytorch=1.4.0=py3.6_cuda10.1.243_cudnn7.6.3_0" \
    "torchvision=0.5.0=py36_cu101" \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4



RUN conda install -y \
  ipython==6.5.0 \
  matplotlib==3.0.3 \
  plac==0.9.6 \
  py==1.6.0 \
  scipy==1.3.1 \
  tqdm==4.36.1 \
  pathlib==1.0.1 \
  seaborn==0.10.0 \
  scikit-learn==0.22.1 \
  scikit-image==0.16.2 \
 && conda clean -ya

RUN conda install -c conda-forge jupyterlab && conda clean -ya

RUN pip install runstats==1.8.0 \
  fire==0.2.1 \
  visdom==0.1.8.9 \
  parse==1.12.1



# Set the default command to python3
CMD ["bash"]

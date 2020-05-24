FROM nvidia/cuda:10.1-base-ubuntu16.04
LABEL version="1.0"
LABEL description="Build using the command \
  'docker build -t epflvil/xtconsistency:latest .'"

ARG DEFAULT_GIT_BRANCH=master
ARG DEFAULT_GIT_REPO=git@github.com:EPFL-VIL/XTConsistency.git
ARG GITHUB_DEPLOY_KEY_PATH=docker_key
ARG GITHUB_DEPLOY_KEY
ARG GITHUB_DEPLOY_KEY_PUBLIC

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    sudo \
    git \
    unzip \
    bzip2 \
    libx11-6 \
    nano \
    screen \
    gcc \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /root/.ssh
RUN echo "DEPLOY" "${GITHUB_DEPLOY_KEY}"
RUN echo "DEPLOY" "${GITHUB_DEPLOY_KEY_PUBLIC}"
RUN echo "${GITHUB_DEPLOY_KEY}" > /root/.ssh/id_rsa
RUN echo "${GITHUB_DEPLOY_KEY_PUBLIC}" > /root/.ssh/id_rsa.pub
RUN chmod 600 /root/.ssh/id_rsa
RUN cat /root/.ssh/id_rsa*
RUN eval $(ssh-agent) && \
    ssh-add /root/.ssh/id_rsa && \
    ssh-keyscan -H github.com >> /etc/ssh/ssh_known_hosts
RUN git clone --single-branch --branch "${DEFAULT_GIT_BRANCH}" "${DEFAULT_GIT_REPO}" /app

#############################
# Pull code
#############################
# RUN mkdir /app
WORKDIR /app

RUN cd /app && git config core.filemode false
RUN chmod -R 777 /app


#############################
# Create non-root user
#############################
# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user


#############################
# Create conda environment
#############################
# Install Miniconda
RUN curl -Lso ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
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


#############################
# Python packages
#############################
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    "pytorch=1.4.0" \
    "torchvision=0.5.0" \
  && conda clean -ya
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

  
###############################################
# Default command and environment variables
###############################################
RUN sudo touch /root/.bashrc && sudo chmod 770 /root/.bashrc
RUN echo export PATH="\$PATH:"$PATH >> /tmp/.bashrc
RUN sudo su -c 'cat /tmp/.bashrc >> /root/.bashrc' && rm /tmp/.bashrc

# Set the default command to bash
CMD ["bash"]

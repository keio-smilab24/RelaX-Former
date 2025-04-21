FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ARG PYTHON_VERSION=3.8.10
ENV TZ=Asia/Tokyo

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        # for usability
        tzdata default-jre \
        # developing tools
        vim wget curl git cmake bash-completion software-properties-common \
        # for Python and pip(from https://devguide.python.org/setup/#linux)
        build-essential gdb lcov pkg-config \
        libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
        libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
        lzma lzma-dev tk-dev uuid-dev zlib1g-dev \
        # for poetry
        python3.8 libreadline-dev llvm libncursesw5-dev xz-utils python-openssl python3-distutils \
    # clean apt cache
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && echo 'source /etc/bash_completion\n' >> /root/.bashrc

# poetry install (use 3.8 to install latest poetry)
RUN curl -sSL https://install.python-poetry.org | python3.8 -
ENV PATH /root/.local/bin:$PATH

# pyenv install
RUN git clone https://github.com/pyenv/pyenv.git /root/.pyenv \
    && echo 'PYENV_ROOT=$HOME/.pyenv\nPATH=$PYENV_ROOT/bin:$PATH\neval "$(pyenv init -)"\neval "$(pyenv init --path)"\n' >> /root/.activate_pyenv \
    && echo 'source /root/.activate_pyenv\n' >> /root/.bashrc

WORKDIR /src
COPY pyproject.toml poetry.lock ./

# poetry modules install
RUN . /root/.activate_pyenv \
    && CONFIGURE_OPTS=--enable-shared pyenv install ${PYTHON_VERSION} \
    && pyenv shell ${PYTHON_VERSION} \
    && poetry env use ${PYTHON_VERSION} \
    && poetry install

# install faiss(developed by Meta Research) for fast oneshot server
RUN . /root/.activate_pyenv \
    && export DEBIAN_FRONTEND=noninteractive \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - \
        | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/nullsoftware-properties-common \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        # install latest cmake
        cmake \
        # depends from faiss
        intel-mkl swig \
        # depends from faiss python-wrapper
        libgflags-dev \
    # clean apt cache
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    # clone and build faiss
    && git clone https://github.com/facebookresearch/faiss.git \
    && mkdir faiss/build \
    && cd faiss/build \
    && poetry run cmake .. "-DCMAKE_CUDA_ARCHITECTURES=75;80;86" \
    && poetry run make -j faiss swigfaiss \
    # install faiss python-wrapper
    && cd /src/faiss/build/faiss/python \
    && poetry run python setup.py install \
    # remove unused files
    && cd /src \
    && rm /src/faiss -r


# download caches
COPY ./src /src/src
RUN . /root/.activate_pyenv \
    && cd src \
    && poetry run python -c "import stanford_parser_process; stanford_parser_process.StanfordParserProcess()" \
    && poetry run python -c "import clip; from main import parse_args; clip.load(parse_args(['mode']).clip_base_model)" \
    && rm /src/src -rf

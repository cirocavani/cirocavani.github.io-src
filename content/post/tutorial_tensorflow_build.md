+++
date = "2016-09-08T22:06:49-03:00"
draft = false
slug = "compilacao-do-tensorflow-0.10-para-linux-com-gpu"
tags = ["Tutorial", "TensorFlow", "Docker", "Ubuntu"]
title = "Compilação do TensorFlow 0.10 para Linux (com GPU)"
+++

Esse tutorial é sobre a construção do pacote do TensorFlow 0.10 para Linux com suporte a GPU. Para esse procedimento é usado o Docker com uma imagem do Ubuntu 16.04, GCC 5.4, Python 2.7, Cuda 8.0 (RC) e cuDNN 5.1. A motivação desse trabalho é usar o TensorFlow com as novas gerações de GPUs da Nvidia ([Pascal](https://developer.nvidia.com/pascal)). Um segundo objetivo é a criação de um pacote do TensorFlow com capacidades específicas (por exemplo, um "Compute Capability" específico).

O procedimento também está disponível como um script para Docker (ainda é necessário fazer o download do Cuda manualmente).

https://github.com/cirocavani/tensorflow-build

...

## Compilação

O procedimento consiste em:

1. Instalar o Cuda 8.0rc com o patch para GCC 5.4
2. Instalar o cuDNN 5.1 para Cuda 8.0
3. Instalar o Java 8
4. Instalar o Bazel 0.3
5. Construir TensorFlow 0.10

O resultado é o pacote do TensorFlow para Python 2 e Linux (com GPU):

    tensorflow-0.10.0-py2-none-linux_x86_64.whl

Baseado na documentação:

https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#installing-from-sources

Um procedimento alternativo:

https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/tools/docker/Dockerfile.devel-gpu

### Download do Cuda 8.0rc, cuDNN 5.1

É necessário o download dos pacotes:

    cuda_8.0.27_linux.run
    cuda_8.0.27.1_linux.run
    cudnn-8.0-linux-x64-v5.1.tgz

Esses pacotes devem ser colocados na pasta `build_deps/`.

...

No momento, a versão mais recente do Cuda é a 8.0 RC e só está disponível para download para membros do [Accelerated Computing Developer Program](https://developer.nvidia.com/accelerated-computing-developer) no site da Nvidia (o cadastro é gratuito).

https://developer.nvidia.com/cuda-release-candidate-download

> Select Target Platform:
>
>     Operating System = Linux
>     Architecture = x86_64
>     Distribution = Ubuntu
>     Version = 16.04
>     Installer Type = runfile (local)
>
> Download:
>
> * **Base Installer** - `cuda_8.0.27_linux.run`
> * **Patch 1** - `cuda_8.0.27.1_linux.run`

https://developer.nvidia.com/rdp/cudnn-download

> Selecione:
>
>     1. I Agree To the Terms of the cuDNN Software License Agreement
>     2. Download cuDNN v5.1 (August 10, 2016), for CUDA 8.0 RC
>     3. cuDNN v5.1 Library for Linux
>
> Download:
>
> * `cudnn-8.0-linux-x64-v5.1.tgz`.


### Setup inicial no Docker para Ubuntu 16.04

Download dos demais pacotes necessários para o build:

```sh
cd build_deps

curl -k -L \
  -H "Cookie: oraclelicense=accept-securebackup-cookie" \
  -O http://download.oracle.com/otn-pub/java/jdk/8u102-b14/jdk-8u102-linux-x64.tar.gz

curl -k -L \
  -O https://github.com/bazelbuild/bazel/releases/download/0.3.1/bazel-0.3.1-installer-linux-x86_64.sh

chmod +x cuda_8.0.27_linux.run
chmod +x cuda_8.0.27.1_linux.run
chmod +x bazel-0.3.1-installer-linux-x86_64.sh

cd ..
```

Criação do Container com as dependências:

```sh
docker create -t --name=tensorflow_build ubuntu:16.04
docker cp build_deps tensorflow_build:/
```

Execução do Shell no Container:

```sh
docker start tensorflow_build
docker exec -i -t tensorflow_build /bin/bash
```

Setup do Container:

```sh
echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
echo 'APT::Install-Recommends "0";' > 01norecommend
mv 01norecommend /etc/apt/apt.conf.d

apt-get update
apt-get upgrade -y

apt-get install -y \
    build-essential \
    python-dev \
    python-wheel \
    python-setuptools \
    python-numpy \
    swig \
    zlib1g-dev \
    unzip \
    file \
    git \
    ca-certificates \
    rsync
```

### Instalação do Cuda 8.0rc e cuDNN 5.1

(comandos a serem executados dentro do container)

```sh
/build_deps/cuda_8.0.27_linux.run --silent --toolkit --override

/build_deps/cuda_8.0.27.1_linux.run --silent --accept-eula

tar zxf /build_deps/cudnn-8.0-linux-x64-v5.1.tgz \
    -C /usr/local/cuda-8.0 --strip-components=1
```

### Instalação do Java 8

(comandos a serem executados dentro do container)

```sh
tar zxf /build_deps/jdk-8u102-linux-x64.tar.gz -C /opt --no-same-owner

echo 'export JAVA_HOME=/opt/jdk1.8.0_102' > /etc/profile.d/java.sh
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> /etc/profile.d/java.sh
chmod a+x /etc/profile.d/java.sh

source /etc/profile.d/java.sh
```

### Instalação do Bazel 0.3

(comandos a serem executados dentro do container)

```sh
/build_deps/bazel-0.3.1-installer-linux-x86_64.sh --prefix=/opt/bazel-0.3.1

echo 'export PATH=$PATH:/opt/bazel-0.3.1/bin' > /etc/profile.d/bazel.sh
chmod a+x /etc/profile.d/bazel.sh

source /etc/profile.d/bazel.sh
```

### Construção do TensorFlow 0.10

Considerações:

* Configuração da GPU

    É necessário definir qual "Compute Capability" o binário do TensorFlow vai suportar.

    https://developer.nvidia.com/cuda-gpus

    Por exemplo:

    A GeForce GT 740M tem Compute Capability 3.0

        export TF_CUDA_COMPUTE_CAPABILITIES=3.0

* Uso de Memória

    O build executa várias tarefas em paralelo e o consumo de memória pode aumentar rapidamente.

    Para limitar o número de execuções paralelas é usada a opção `-j 4` no build.

    Em um notebook com 8 cores (HT), 8G de memória é insuficiente.


(comandos a serem executados dentro do container)

```sh
useradd -m tensorflow
passwd -d tensorflow

su - tensorflow

git clone https://github.com/tensorflow/tensorflow.git -b r0.10 ~/tensorflow-0.10

cd ~/tensorflow-0.10

export PYTHON_BIN_PATH=/usr/bin/python
export TF_NEED_GCP=0
export TF_NEED_CUDA=1
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export TF_CUDA_VERSION=8.0
export CUDA_TOOLKIT_PATH=/usr/local/cuda-8.0
export TF_CUDNN_VERSION=5
export CUDNN_INSTALL_PATH=/usr/local/cuda-8.0
export TF_CUDA_COMPUTE_CAPABILITIES=3.0
./configure

bazel build -j 4 -c opt --config=cuda \
    //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package $HOME

mv ~/tensorflow-0.10.0-py2-none-{any,linux_x86_64}.whl

# saindo su
exit

# saindo do container
exit
```
...

Para baixar o pacote (fora do container):

```sh
docker cp \
    tensorflow_build:/home/tensorflow/tensorflow-0.10.0-py2-none-linux_x86_64.whl \
    .
```

## Conclusão

O procedimento de build do TensorFlow não é complicado, mas pequenas variações podem atingir alguns bugs do build ([exemplo](https://github.com/tensorflow/tensorflow/issues/3985)). Com um script bem definido, fica fácil criar o pacote do TensorFlow.

Com esse pacote, é possível usar o TensorFlow nas GPUs mais recentes da Nvidia.

No próximo artigo será um tutorial de como configurar um ambiente de desenvolvimento com Jupyter.

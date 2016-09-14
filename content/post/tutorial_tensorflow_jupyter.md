+++
date = "2016-09-13T21:13:26-03:00"
draft = false
slug = "tensorflow-no-jupyter-com-notebooks"
tags = ["TensorFlow", "Jupyter", "Tutorial"]
title = "TensorFlow no Jupyter (com notebooks)"
+++

Esse tutorial é sobre o TensorFlow no Jupyter. A princípio, esse projeto pode ser usado para instalar automaticamente o Jupyter Notebook configurado com TensorFlow 0.10 e alguns notebooks de exemplo (tutoriais do TensorFlow). Outro objetivo é servir como base para criação de configurações customizadas isoladas (exemplo um ambiente extra para testar com TensorFlow GPU Python 3 com CUDA 8). O Jupyter é uma ferramenta excelente para testar ideias e prototipar rapidamente com TensorFlow.

**Projeto**

https://github.com/cirocavani/tensorflow-jupyter


Esse artigo consiste em:

1. o procedimento de instalação básico
2. a descrição dos notebooks de exemplo
3. a explicação de como funciona a instalação

## Instalação

    git clone https://github.com/cirocavani/tensorflow-jupyter.git
    cd tensorflow-jupyter

    #bin/setup-linux
    bin/setup-mac

Comandos:

    bin/jupyter

Inicializa o Jupyter que já tem o kernel do TensorFlow configurado.

Acesso em [http://localhost:8888/](http://localhost:8888/).


    bin/tensorboard

Inicializa a ferramenta de visualização do TensorFlow, mostra grafo de execução, valores de medições do treinamento.

Acesso em [http://localhost:6006/](http://localhost:6006/).


## Notebooks Exemplo

Os notebooks exemplo são baseados nos tutorias disponiveis no site do TensorFlow. Os tutoriais originais estão referenciados no início do notebook. O código de alguns tutoriais foi alterado para usar algumas funcionalidades mais "reais" (por exemplo: leitura de CSV em batch).


**0 - First Run**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Examples/00%20-%20First%20Run.ipynb)

Hello World com TensorFlow.


**1 - Linear Regression**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Examples/01%20-%20Linear%20Regression.ipynb)

Nesse exemplo, é feito uma regressão linear para o fit de uma reta em dados gerados sinteticamente pela função y = 0.1x + 0.3, ou seja, o TensorFlow aprende os parâmetros 0.1 e 0.3 de um dataset ruidoso.


**2 - MNIST, Softmax Regression**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Examples/02%20-%20MNIST%2C%20Softmax%20Regression.ipynb)

Nesse exemplo, é feito um classificador com uma regressão softmax para identificação de dígitos 0-9 em uma imagem. Dado na entrada uma imagem de 28x28 pixels de um dígito manuscrito, o classificador retorna 10 valores, cada um indicando a "probabilidade" de ser um dos dígitos que a variável representa. A acurácia é de 92%.


**3 - MNIST, Convolutional Network**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Examples/03%20-%20MNIST%2C%20Convolutional%20Network.ipynb)

Nesse exemplo, é feito um classificador com uma rede neural convolutiva para identificação de dígitos 0-9 em uma imagem. Dado na entrada uma imagem de 28x28 pixels de um dígito manuscrito, o classificador retorna 10 valores, cada um indicando a "probabilidade" de ser um dos dígitos que a variável representa. A acurácia é de 99%.

A rede é formada por duas camadas de convolução, uma camada toda conectada, uma camada de dropout e uma camada de regressão softmax.


**4 - MNIST, Feed-forward NN with Log**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Examples/04%20-%20MNIST%2C%20Feed-forward%20NN%20with%20Log.ipynb)

Nesse exemplo, é feito um classificador com uma rede neural feed-forward para identificação de dígitos 0-9 em uma imagem. Dado na entrada uma imagem de 28x28 pixels de um dígito manuscrito, o classificador retorna 10 valores, cada um indicando a "probabilidade" de ser um dos dígitos que a variável representa. A acurácia é de 99%.

A rede é formada por duas camadas toda conectada e uma camada de regressão softmax.

O modelo treinado nesse notebook pode ser visualizado no TensorBoard.


**5 - Iris, DNN Classifier (tf.contrib.learn)**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Examples/05%20-%20Iris%2C%20DNN%20Classifier%20%28tf.contrib.learn%29.ipynb)

Nesse exemplo, é feito um classificador com uma rede neural para classificação de 3 espécies de flor. Dada na entrada as medidas da sépala e da pétala, o classificador retorna a espécie 0 setosa, 1 versicolor e 2 virginica. A acurácia é de 97%.

A rede é formada por 5 camadas.

O modelo treinado nesse notebook pode ser visualizado no TensorBoard.


**6 - Iris, DNN Classifier with Log (tf.contrib.learn)**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Examples/06%20-%20Iris%2C%20DNN%20Classifier%20with%20Log%20%28tf.contrib.learn%29.ipynb)

Nesse exemplo, é feito um classificador com uma rede neural para classificação de 3 espécies de flor. Dada na entrada as medidas da sépala e da pétala, o classificador retorna a espécie 0 setosa, 1 versicolor e 2 virginica. A acurácia é de 97%.

A rede é formada por 5 camadas e é feito o monitoramento de métricas que podem ser visualizadas no log do notebook e no TensorBoard.

O modelo treinado nesse notebook pode ser visualizado no TensorBoard.


**7 - Reading CSV**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Examples/07%20-%20Reading%20CSV.ipynb)

Nesse exemplo, é feito o pipeline para leitura de dados de um arquivo CSV. O arquivo usado nesse estudo é o mesmo do Census.


**8 - Census, Logistic Regression Classifier (tf.contrib.learn)**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Examples/08%20-%20Census%2C%20Logistic%20Regression%20Classifier%20%28tf.contrib.learn%29.ipynb)

Nesse exemplo, é feito um classificador com uma regressão logística para classificação de rendimento maior que 50 mil dólares. Dada na entrada as informações do Census, o classificador retorna 1 mais de 50 mil e 0 menos de 50 mil. A acurácia é de 87%.

O modelo treinado nesse notebook pode ser visualizado no TensorBoard.


**9 - Census, Combined Classifier (tf.contrib.learn)**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Examples/09%20-%20Census%2C%20Combined%20Classifier%20%28tf.contrib.learn%29.ipynb)

Nesse exemplo, é feito um classificador com a combinação de uma rede neural e uma regressão logística (treinadas em conjunto) para classificação de rendimento maior que 50 mil dólares. Dada na entrada as informações do Census, o classificador retorna 1 mais de 50 mil e 0 menos de 50 mil. A acurácia é de 93%.

O modelo treinado nesse notebook pode ser visualizado no TensorBoard.


## Funcionamento

>IMPORTANTE:
>
> Essa é a descrição de como é feita a configuração do projeto, contudo esse procedimento já está definido no comando `bin/setup-linux` (ou `bin/setup-mac`) que deve ser executado ao invés desse procedimento.
>
> Esse passo a passo é para ajudar na customização do Projeto.

O procedimento de instalação consiste em:

1. Instalar o Python 2.7 com `miniconda` (Linux ou Mac)
2. Instalar o Jupyter Notebook 4.2 no *environment* `default` do `conda2`
3. Instalar o TensorFlow 0.10 em um *environment* próprio (para Python 2.7)
4. Instalar o kernel do Python no *environment* do TensorFlow
5. Configurar o kernel no Jupyter que é executado no *environment* do TensorFlow

A estrutura do projeto será:

* `deps/conda2`: instalação do Python 2.7 (`miniconda`)
* `deps/tensorflow-0.10`: instalação do TensorFlow 0.10 (*environment* isolado)
* `data/kernels/tensorflow-0.10/kernel.json`: configuração do kernel no Jupyter para o TensorFlow

Ao final desse procedimento, a execução do Jupyter consiste de (`bin/jupyter`):

    export JUPYTER_DATA_DIR=`pwd`/data
    deps/conda2/bin/jupyter notebook --no-browser --notebook-dir=`pwd`/workspace

Para a criação de uma customização, os passos 3, 4 e 5 devem ser ajustados para um novo *environment* com configuração customizada.


### Instalação do Python

A instalação do Python é feita usando o `miniconda` para versão 2.7 (para Linux ou Mac).

Os comandos de Python e Conda ficam disponíveis na pasta `deps/conda2/bin`.

http://conda.pydata.org/miniconda.html

(Linux)

    curl -k -L \
        -O https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh

    chmod +x Miniconda2-latest-Linux-x86_64.sh

    ./Miniconda2-latest-Linux-x86_64.sh -b -f -p deps/conda2


### Instalação do Jupyter

O Jupyter tem um meta pacote que depende de todos os componentes, incluindo o Notebook.

O comando do Jupyter fica disponível em `deps/conda2/bin/jupyter`.

http://jupyter.readthedocs.io/en/latest/install.html

https://pypi.python.org/pypi/jupyter

    deps/conda2/bin/pip install --upgrade jupyter


### Instalação do TensorFlow

O TensorFlow é distribuído como um pacote Wheel e é instalado em um *environment* isolado criado no `miniconda`.

O comando do TensorBoard fica disponível em `deps/tensorflow-0.10/bin/tensorboard`.

https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#anaconda-installation

(Linux)

    deps/conda2/bin/conda create -y -p deps/tensorflow-0.10 python=2.7

    deps/tensorflow-0.10/bin/pip install \
        https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl

### Instalação do Kernel para o TensorFlow

No *environment* do TensorFlow é instalado o kernel Python que possibilita a conexão a partir do Jupyter (Notebook). Isso torna possível escrever código Python que é executado dentro desse *environment* isolado.

http://ipython.readthedocs.io/en/stable/install/kernel_install.html

https://pypi.python.org/pypi/ipykernel

    deps/tensorflow-0.10/bin/pip install ipykernel

### Configuração do Kernel para o TensorFlow

O Jupyter é configurado com o comando que executa o kernel do Python dentro do *environment* que tem o TensorFlow. O kernel é responsável por receber requisições do servidor do Jupyter e executar código Python no processo em que está executando. Esse processo é executado somente com os pacotes do próprio *environment* (isolamento) e pacotes adicionais devem ser instalados nesse *environment* sem conflito com outros *environments*.

https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernelspecs

    mkdir -p data/kernels/tensorflow-0.10-py2

    echo "{
     \"display_name\": \"TensorFlow 0.10 (CPU, Python 2)\",
     \"language\": \"python\",
     \"argv\": [
      \"`pwd`/deps/tensorflow-0.10/bin/python\",
      \"-c\",
      \"from ipykernel.kernelapp import main; main()\",
      \"-f\",
      \"{connection_file}\"
     ]
    }" > data/kernels/tensorflow-0.10-py2/kernel.json


## Conclusão

O Jupyter é uma excelente ferramenta para exploração de ideias e desenvolvimento de código rápido. A facilidade de visualização e execução independente de células é muito prático. O desenvolvimento de aplicações mais complexas e com código mais estruturado já não é muito favorável.

O TensorFlow é uma ferramenta sofisticada para desenvolvimento de algoritmos inteligentes. Algumas APIs podem ser complexas e de difícil entendimento, algumas vezes bem documentadas e outras não. A comunidade é muito engajada e o Google vem produzindo tutoriais, documentação e modelos muito úteis para o aprendizado.

Aprender TensorFlow no Jupyter é o melhor caminho e essa é a proposta desse Projeto.

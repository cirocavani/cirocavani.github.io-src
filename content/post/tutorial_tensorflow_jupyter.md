+++
date = "2016-09-09T21:13:26-03:00"
draft = true
slug = "tensorflow-no-jupyter-com-notebooks"
tags = ["TensorFlow", "Jupyter", "Tutorial"]
title = "TensorFlow no Jupyter (com notebooks)"
+++

Esse tutorial é sobre o TensorFlow no Jupyter.

**Projeto**

https://github.com/cirocavani/tensorflow-jupyter


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

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Example%2000%20-%20First%20Run.ipynb)

Hello World com TensorFlow.


**1 - Linear Regression**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Example%2001%20-%20Linear%20Regression.ipynb)

Nesse exemplo, é feito uma regressão linear para o fit de uma reta em dados gerados sinteticamente pela função y = 0.1x + 0.3, ou seja, o TensorFlow aprende os parâmetros 0.1 e 0.3 de um dataset ruidoso.


**2 - MNIST, Softmax Regression**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Example%2002%20-%20MNIST%2C%20Softmax%20Regression.ipynb)

Nesse exemplo, é feito um classificador com uma regressão softmax para identificação de dígitos 0-9 em uma imagem. Dado na entrada uma imagem de 28x28 pixels de um dígito manuscrito, o classificador retorna 10 valores, cada um indicando a "probabilidade" de ser um dos dígitos que a variável representa. A acurácia é de 92%.


**3 - MNIST, Convolutional Network**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Example%2003%20-%20MNIST%2C%20Convolutional%20Network.ipynb)

Nesse exemplo, é feito um classificador com uma rede neural convolutiva para identificação de dígitos 0-9 em uma imagem. Dado na entrada uma imagem de 28x28 pixels de um dígito manuscrito, o classificador retorna 10 valores, cada um indicando a "probabilidade" de ser um dos dígitos que a variável representa. A acurácia é de 99%.

A rede é formada por duas camadas de convolução, uma camada toda conectada, uma camada de dropout e uma camada de regressão softmax.


**4 - MNIST, Feed-forward NN with Log**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Example%2004%20-%20MNIST%2C%20Feed-forward%20NN%20with%20Log.ipynb)

Nesse exemplo, é feito um classificador com uma rede neural feed-forward para identificação de dígitos 0-9 em uma imagem. Dado na entrada uma imagem de 28x28 pixels de um dígito manuscrito, o classificador retorna 10 valores, cada um indicando a "probabilidade" de ser um dos dígitos que a variável representa. A acurácia é de 99%.

A rede é formada por duas camadas toda conectada e uma camada de regressão softmax.

O modelo treinado nesse notebook pode ser visualizado no TensorBoard.


**5 - Iris, DNN Classifier (tf.contrib.learn)**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Example%2005%20-%20Iris%2C%20DNN%20Classifier%20%28tf.contrib.learn%29.ipynb)

Nesse exemplo, é feito um classificador com uma rede neural para classificação de 3 espécies de flor. Dada na entrada as medidas da sépala e da pétala, o classificador retorna a espécie 0 setosa, 1 versicolor e 2 virginica. A acurácia é de 97%.

A rede é formada por 5 camadas.

O modelo treinado nesse notebook pode ser visualizado no TensorBoard.


**6 - Iris, DNN Classifier with Log (tf.contrib.learn)**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Example%2006%20-%20Iris%2C%20DNN%20Classifier%20with%20Log%20%28tf.contrib.learn%29.ipynb)

Nesse exemplo, é feito um classificador com uma rede neural para classificação de 3 espécies de flor. Dada na entrada as medidas da sépala e da pétala, o classificador retorna a espécie 0 setosa, 1 versicolor e 2 virginica. A acurácia é de 97%.

A rede é formada por 5 camadas e é feito o monitoramento de métricas que podem ser visualizadas no log do notebook e no TensorBoard.

O modelo treinado nesse notebook pode ser visualizado no TensorBoard.


**7 - Reading CSV**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Example%2007%20-%20Reading%20CSV.ipynb)

Nesse exemplo, é feito o pipeline para leitura de dados de um arquivo CSV. O arquivo usado nesse estudo é o mesmo do Census.


**8 - Census, Logistic Regression Classifier (tf.contrib.learn)**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Example%2008%20-%20Census%2C%20Logistic%20Regression%20Classifier%20%28tf.contrib.learn%29.ipynb)

Nesse exemplo, é feito um classificador com uma regressão logística para classificação de rendimento maior que 50 mil dólares. Dada na entrada as informações do Census, o classificador retorna 1 mais de 50 mil e 0 menos de 50 mil. A acurácia é de 87%.

O modelo treinado nesse notebook pode ser visualizado no TensorBoard.


**9 - Census, Combined Classifier (tf.contrib.learn)**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Example%2009%20-%20Census%2C%20Combined%20Classifier%20%28tf.contrib.learn%29.ipynb)

Nesse exemplo, é feito um classificador com a combinação de uma rede neural e uma regressão logística (treinadas em conjunto) para classificação de rendimento maior que 50 mil dólares. Dada na entrada as informações do Census, o classificador retorna 1 mais de 50 mil e 0 menos de 50 mil. A acurácia é de 93%.

O modelo treinado nesse notebook pode ser visualizado no TensorBoard.


## Configuração

TBD

## Conclusão

TBD

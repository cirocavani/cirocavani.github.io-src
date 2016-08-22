+++
date = "2015-09-24T19:44:08-03:00"
draft = false
slug = "otimizacao-dos-parametros-do-spark-als-collaborative-filtering-usando-moe"
tags = ["Tutorial", "Otimização", "Sistema de Recomendação", "Algoritmos", "Spark"]
title = "Otimização dos parâmetros do Spark ALS (Collaborative Filtering) usando MOE"
+++

Esse tutorial é sobre otimização de parâmetros em modelos de machine learning. Para esse tutorial, a ferramenta utilizada é o MOE, Metric Optimization Engine, desenvolvido pelo Yelp que implementa o algoritmo de busca usando Gaussian Process. O algoritmo escolhido para ter os parâmetros otimizados é o Collaborative Filtering baseado na fatoração da matriz de preferências. De forma genérica, esse é um processo que pode ser facilmente adaptado para outros algoritmos e permite sistematizar a árdua tarefa de escolher os melhores parâmetros para um modelo.

http://yelp.github.io/MOE/

> MOE (Metric Optimization Engine) is an efficient way to optimize a system’s parameters, when evaluating parameters is time-consuming or expensive.
>
> How does MOE work?
>
> 1. Build a Gaussian Process (GP) with the historical data
> 2. Optimize the hyperparameters of the Gaussian Process
> 3. Find the point(s) of highest Expected Improvement (EI)
> 4. Return the point(s) to sample, then repeat

Primeiramente, é feita a instalação do MOE. Nesse processo, é necessário configurar o ambiente para compilar as dependências do projeto e o código que é composto por Python e C++. No final desse procedimento, o serviço do MOE estará disponível como um servidor REST e a API Python que pode ser usada para definir o procedimento de otimização.

O procedimento de instalação em detalhes é descrito aqui:

http://yelp.github.io/MOE/install.html#install-from-source

```sh

mkdir grandesdados-opt
cd grandesdados-opt

virtualenv --no-site-packages --python=python2.7 moe-env

> Running virtualenv with interpreter /usr/bin/python2.7
> New python executable in moe-env/bin/python2.7
> Also creating executable in moe-env/bin/python
> Installing setuptools, pip, wheel...done.

source moe-env/bin/activate

git clone https://github.com/Yelp/MOE.git
cd MOE

pip install -r requirements.txt

> (...)
> Successfully installed (...)

python setup.py install

> (...)

pserve --reload development.ini

> (...)
> Starting server in PID 23232.
> serving on 0.0.0.0:6543 view at http://127.0.0.1:6543

# (nesse momento, esse terminal fica 'preso' mostrando o log do servidor do MOE)

```

O próximo passo é instalar o algoritmo que tem parâmetros que precisam ser otimizados.

Nesse tutorial, será usado o algoritmo de Collaborative Filtering baseado na fatoração da matriz de preferências que gera um vetor para cada usuário e item da matriz original. Nesse algoritmo, os parâmetros são a dimensão do vetor a ser gerado (fatores latentes), o número de iterações para fatoração da matriz e o parâmetro de regularização usado na fatoração.

O DataSet usado é um sample MovieLens que já vem na distribuição do Spark. São 1501 ratings, 30 usuários e 100 filmes.

O código pode ser análisado aqui:

https://github.com/apache/spark/blob/v1.5.0/examples/src/main/scala/org/apache/spark/examples/ml/MovieLensALS.scala

(procedimento na mesma pasta anterior `grandesdados-opt`)

```sh
curl -L -O http://ftp.unicamp.br/pub/apache/spark/spark-1.5.0/spark-1.5.0-bin-hadoop2.6.tgz
tar zxf spark-1.5.0-bin-hadoop2.6.tgz
cd spark-1.5.0-bin-hadoop2.6/

cp conf/log4j.properties{.template,}
sed -i s/log4j\.rootCategory\=INFO/log4j\.rootCategory\=ERROR/1 conf/log4j.properties

echo "spark.ui.showConsoleProgress=false" > conf/spark-defaults.conf

```

Executando o exemplo do MovieLens (a saída são os parâmetros):

```sh

./bin/run-example ml.MovieLensALS

> Error: Missing option --ratings
> Error: Missing option --movies
> MovieLensALS: an example app for ALS on MovieLens data.
> Usage: MovieLensALS [options]
>
>   --ratings <value>
>         path to a MovieLens dataset of ratings
>   --movies <value>
>         path to a MovieLens dataset of movies
>   --rank <value>
>         rank, default: 10
>   --maxIter <value>
>         max number of iterations, default: 10
>   --regParam <value>
>         regularization parameter, default: 0.1
>   --numBlocks <value>
>         number of blocks, default: 10
>
> Example command line to run this app:
>
>  bin/spark-submit --class org.apache.spark.examples.ml.MovieLensALS \
>   examples/target/scala-*/spark-examples-*.jar \
>   --rank 10 --maxIter 15 --regParam 0.1 \
>   --movies data/mllib/als/sample_movielens_movies.txt \
>   --ratings data/mllib/als/sample_movielens_ratings.txt

```

Fazendo uma execução:

```sh

time ./bin/run-example ml.MovieLensALS \
--rank 10 --maxIter 15 --regParam 0.1 \
--movies data/mllib/als/sample_movielens_movies.txt \
--ratings data/mllib/als/sample_movielens_ratings.txt

> Got 1501 ratings from 30 users on 100 movies.                                   
> Training: 1169, test: 332.
> Test RMSE = 0.9815785141168548.                                                 
> Found 0 false positives                                                         
>
> real	0m22.441s
> user	0m56.320s
> sys	0m1.847s

```

Para fazer a otimização, o MOE requer que o problema seja modelado como uma função do vetor de parâmetros para um valor escalar. O objetivo da ferramenta é minimizar essa função.

Para o problema do ALS, por simplicidade, vamos aproveitar que o exemplo já calcula o RMSE e usar a função que mapeia o vetor do Número de Fatores Latentes, Número de Iterações e Regularização para o RMSE. Faz sentido o objetivo ser minimizar o RMSE.

Na pasta `grandesdados-opt`, crie o arquivo `func.sh` que mapeia a função desejada:

```sh
#!/bin/bash

cd spark-1.5.0-bin-hadoop2.6/

./bin/run-example ml.MovieLensALS \
--rank $1 --maxIter $2 --regParam $3 \
--movies data/mllib/als/sample_movielens_movies.txt \
--ratings data/mllib/als/sample_movielens_ratings.txt 2>&1 \
| sed -n 's/\(Test RMSE =\) \([0-9]*\.[0-9]*\)\./\2/p'
```

Dessa forma, teremos:

```sh
chmod +x func.sh

./func.sh 10 15 0.1

> 0.9815785141168546
```

Agora podemos definir o procedimento de otimização como um experimento do MOE.

O experimento é criado com o domínio dos parâmetros que estamos querendo otimizar. Dado que estamos trabalhando com um DataSet limitado, podemos restringir os valores.

Nesse exemplo, estamos usando:

* Número de Fatores Latentes: entre 5 e 50
* Número de Iterações da Fatoração: entre 5 e 20
* Regularização da Fatoração: entre 0.001 e 1

Vamos assumir como primeiro ponto 'ótimo' o que vem documentado no exemplo, ou seja, 10, 15 e 0,1.

A busca por uma solução muito boa pode envolver muitas iterações do processo de otimização, nesse exemplo vamos usar 20, mas poderia ser 100 ou 400 para uma busca mais completa.

Na pasta `grandesdados-opt`, crie o arquivo `opt.py` que define o procedimento de otimização:

```python
import subprocess

from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint

def function_to_minimize(x):
    f = "./func.sh {0:} {1:} {2:}".format(int(x[0]), int(x[1]), x[2])
    print f
    y = subprocess.Popen(f, shell=True, stdout=subprocess.PIPE).stdout.read().strip()
    if y: print y
    return float(y)

if __name__ == '__main__':
    exp = Experiment([[5, 50], [5, 20], [0.001, 1]])

    xmin = []
    ymin = 0.0

    for i in range(20):
        print "Sample {0:}".format(i)
        try:
            x = [10.0, 15.0, 0.1] if i == 0 else gp_next_points(exp)[0]
            y = function_to_minimize(x)
            exp.historical_data.append_sample_points([
                SamplePoint(x, y, 0.05),
            ])
            if not xmin or y < ymin:
                xmin, ymin = x, y
        except ValueError:
            print "error"
        print

    print str(xmin)
    print str(ymin)

```

Por fim, o resultado:
<br/>(necessário estar dentro do virtualenv onde o MOE foi instalado)

```sh
python2 opt.py

> Sample 0
> ./func.sh 10 15 0.1
> 0.9815785141168545
> (...)
> Sample 19
> ./func.sh 16 19 0.41989952735
> error
>
> [46.6604641336, 19.9410400182, 0.0409527639665]
> 0.977384860516

```

Como podemos ver, os parâmetros 46 para Número de Fatores Latentes, 19 para Iterações da Fatoração e 0,041 para Regularização resultaram em um erro menor nesse dataset de exemplo.

Para mais informações sobre otimização usando o MOE, consulte a documentação.

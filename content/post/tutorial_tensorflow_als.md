+++
date = "2017-03-01T11:31:34-03:00"
draft = false
slug = "tensorflow-recomendacao-com-als-collaborative-filtering"
tags = ["TensorFlow", "Spark", "Sistema de Recomendação", "Algoritmos", "Tutorial"]
title = "TensorFlow: Recomendação com ALS (Collaborative Filtering)"

+++

Esse artigo é sobre a análise do ALS implementado no TensorFlow. O ALS é um método para fatoração de matriz usado como algoritmo de *Collaborative Filtering* em Sistemas de Recomendação. A análise consiste no treinamento e *tuning* desse algoritmo e a avaliação do erro final. Para comparação, o mesmo algoritmo é implementado com o Spark. A metodologia usada tem características peculiares de como a Recomendação e o ALS funcionam. O resultado mostra que o Spark tem performance melhor que o TensorFlow no erro final.


**Código**

[Notebook](https://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Recommendation/ALS.ipynb)


## Motivação

Desde que comecei a trabalhar com Recomendação na Globo.com, já coloquei em Produção mais de uma implementação do ALS. É um algoritmo relativamente fácil de entender e que tem excelentes resultados na prática. Atualmente, a implementação que usamos em Produção é a do [Spark 2](https://youtu.be/Q0VXllYilM0). O TensorFlow é uma tecnologia que possibilita a implementação de algoritmos mais sofisticados de Inteligência Artificial que tenho interesse em usar em Produção. Seria ideal que pudesse ser usado nos algoritmos mais comuns que tem boa performance.

Com base nessa ideia, essa análise é uma primeira comparação entre essas duas implementações do TensorFlow e do Spark.

> **IMPORTANTE**
>
> Essa análise foi feita com um dataset pequeno com objetivo de facilitar o desenvolvimento, portanto, os resultados obtidos são apenas para ter uma ideia e não servervem para chegar em *conclusões definitivas* sobre essas implementações.

Tomei conhecimento de que o TensorFlow tinha a implementação do ALS a partir de um vídeo do [TensorFlow Dev Summit](https://events.withgoogle.com/tensorflow-dev-summit/) que ocorreu em 15/Fevereiro (WALS no tempo 2:20):

{{< youtube Tuv5QYKU-MM >}}

## Introdução

A ideia geral é simples:

> Usuários dão rating para alguns filmes e o algoritmo gera uma lista de outros filmes que o usuário também daria um bom rating.
>
> O ALS é um método de fatoração de matriz que é usado para 'completar' os ratings dos filmes que o usuário não deu rating, baseado nos ratings que vários usuários deram aos filmes.
>
> Cada usuário e filme é transformado em um vetor de números (fatores) que são ajustados para representar o interesse do usuário em uma determinada característica de um filme (cada fator é um 'peso' que indica quanto o usuário gosta e quanto o filme oferece). O produto entre os fatores do usuário e os fatores do filme tem que ser 'igual' ao rating que o usuário deu ao filme.
>
> No caso dos filmes que o usuário não deu rating (não viu?), esse produto é o 'rating previsto'. Ordenando todos os ratings previstos, os maiores são usados para recomendação.
>
> Esse é o algoritmo de Collaborative Filtering com ALS.
>
> Esse é o algoritmo que ficou famoso no prêmio Netflix.

Nesse trabalho, a análise do ALS consiste em:

1. Preparação de Dados: ratings do MovieLens, dataset para treinamento, validação e teste
2. Treinamento com TensorFlow: algoritmo que completa a matriz de ratings com ALS do TensorFlow
3. Treinamento com Spark: algoritmo que completa a matriz de ratings com ALS do Spark
4. Seleção de Parâmetros: busca da combinação com menor erro no dataset de validação
5. Comparação: avaliação do erro no dataset de teste da melhor combinação de parâmetros

## Preparação dos Dados

Os dados usados nessa análise são do [MovieLens](https://grouplens.org/datasets/movielens/).

**MovieLens Small**

[ [README](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) ]
[ [ZipFile](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip) ]

> This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 100004 ratings and 1296 tag applications across 9125 movies. These data were created by 671 users between January 09, 1995 and October 16, 2016. This dataset was generated on October 17, 2016.
>
> Users were selected at random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user is represented by an id, and no other information is provided.
>
>The data are contained in the files links.csv, movies.csv, ratings.csv and tags.csv.

...

O dataset consiste de 100.004 ratings registrados por 671 usuários em 9.066 vídeos (o número de vídeos com rating é menor que o número de vídeos com tag, 9.125). Como esperado, a matriz de usuários por vídeos é bastante esparsa: apenas 1,64% de ratings dos 6.083.286 (671 x 9.066) possíveis.

> Diferente desse dataset, em que o número de usuários é bem menor que o número de itens (menor que 1/10), na recomendação da Globo.com normalmente a proporção é inversa, ou seja, muito mais usuários do que itens - nas nossas próprias análises, essa é uma característica relevante.

Esse dataset é bastante pequeno e serve ao propósito de desenvolvido da análise e não para encontrar 'grandes verdades'.

A estratégia é dividir esses dados para treinamento, validação e teste. O dataset de treinamento será usado como os dados que o algoritmo conhece do usuário (e deve aprender sobre). O dataset de validação é para ser usado durante o treinamento para medir a performance do algoritmo, verificar overfitting (ou under) e fazer tuning de parâmetros. O dataset de teste será usado uma única vez para medir o desempenho final do algoritmo com os melhores parâmetros.

O critério usado para dividir os dados é baseado em uma especificidade de Recomendação. No pipeline de Produção, um algoritmo é treinado com os dados históricos e tem sua performance avaliada em tempo real. Desconsiderando o impacto que a própria recomendação possa ter no consumo de itens, essa mesma 'dinâmica temporal' é usada para dividir os dados.

Os ratings são ordenados pelo timestamp em que foram feitos. Os primeiros 80% desses ratings são designados para treinamento / validação e os últimos 20% são designados para teste. Novamente, o primeiro dataset é ordenado e dividido em 80% para treinamento e 20% para validação. A divisão, portanto, fica 64% para treinamento, 16% para validação e 20% para teste.

> Uma variação desse critério: separar por tempo primeiro entre 70% treinamento e 30% validação / teste, depois separar por shuffle 15% de validação e 15% de teste. (Escolhi usar só o critério de tempo porque é mais próximo de Produção)

O dataset de treinamento tem 64.002 ratings, 435 usuários e 5.668 vídeos.

O dataset de validação tem 16.001 ratings, 136 usuários e 4.112 vídeos.

O dataset de teste tem 20.001 ratings, 147 usuários e 4.753 vídeos.

...

A medida de performance usada nesse análise é o RMSE ([Root Mean Square Error](https://en.wikipedia.org/wiki/Root-mean-square_error)) onde o 'erro' é a diferença entre o rating atribuído pelo usuário a um vídeo e o rating calulado pelo produto entre o vetor de fatores desse usuário e o vetor de fatores desse vídeo. O RMSE é uma medida aproximada de quanto o algoritmo pode errar a predição de rating, para mais ou para menos. A expectativa é que esse valor seja muito pequeno para o dataset de treinamento (o ALS minimiza um função similar ao RMSE).

Para efeito de avaliação de performance, temos uma especificidade do ALS. Uma vez que é necessário ter o vetor de fatores tanto do usuário quanto do vídeo para estimar o rating, apenas usuários e vídeos que estão simultaneamente no dataset de treinamento e validação (ou teste) podem ser considerados para o cálculo do RMSE. Nesse caso, estamos avaliando a capacidade de predição do algoritmo e ignorando a cobertura (tanto em usuários ou vídeos).

Apenas um subconjunto do dataset de validação e teste é usado para avaliação.

(Todo o dataset de treinamento pode ser usado na avaliação)

A avaliação com o dataset de validação tem 944 ratings, 23 usuários e 2.424 vídeos.

A avaliação com o dataset de teste tem 278 ratings, 5 usuários e 2.332 vídeos.


## Treinamento com TensorFlow

A implementação do algoritmo foi baseada na documentação da classe WALS do TensorFlow e nos testes dessa classe.

Documentação da implementação do WALS:

[ [GitHub: factorization_ops.py#L53-L166](https://github.com/tensorflow/tensorflow/blob/v1.0.0/tensorflow/contrib/factorization/python/ops/factorization_ops.py#L53-L166) ]

Documentação dos parâmetros do WALS:

[ [GitHub: factorization_ops.py#L181-L214](https://github.com/tensorflow/tensorflow/blob/v1.0.0/tensorflow/contrib/factorization/python/ops/factorization_ops.py#L181-L214) ]

Código do teste da classe WALS:

[ [GitHub: factorization_ops_test.py#L534-L576](https://github.com/tensorflow/tensorflow/blob/v1.0.0/tensorflow/contrib/factorization/python/ops/factorization_ops_test.py#L534-L576) ]

Código da loss function:

[ [GitHub: factorization_ops_test.py#L105-L158](https://github.com/tensorflow/tensorflow/blob/df5d3cd42335e31bccb6c796169d000d73c747d3/tensorflow/contrib/factorization/python/ops/factorization_ops_test.py#L105-L158) ]

Rascunho do algoritmo:

[Notebook](http://nbviewer.jupyter.org/github/cirocavani/tensorflow-jupyter/blob/master/workspace/Recommendation/ALS%20draft.ipynb)

...

O algoritmo é implementado em duas classes:

1. `ALSRecommender`: classe responsável pelo treinamento (recebe um dataset com ratings, calcula os fatores dos usuários e vídeos com o ALS e retorna o modelo com esses fatores)
2. `ALSRecommenderModel`: classe responsável pela inferência (recebe um par usuário e vídeo e retorna a predição do rating ou recebe um usuário e retorna os vídeos com maior rating para esse usuário)


**ALSRecommender**

A classe `ALSRecommender` recebe três parâmetros:

* `num_factors` (default 10): número de fatores em que cada usuário e vídeo devem ser representados (valor muito grande pode resultar em overfitting, muito pequeno em underfitting; custo computacional, tamanho da matriz de usuários e vídeos)
* `num_iters` (default 10): número de repetições do método do ALS (a convergência normalmente é rápida, portando um número muito grande pode não ajudar muito; custo computacional)
* `reg` (default 1e-1): fator de regularização (impacto na convergência, valor muito grande pode resultar em instabilidade e um valor muito pequeno pode resultar em overfitting)

O treinamento é implementado no método `fit` e consiste em três passos: transformação dos dados em matriz esparsa, criação do ALS e execução do ALS.

No final é retornanda uma instância do `ALSRecommenderModel` com a matriz de usuários e matriz de vídeos (itens).

```python
def fit(self, dataset, verbose=False):
    with tf.Graph().as_default(), tf.Session() as sess:
        input_matrix, mapping = self.sparse_input(dataset)
        model = self.als_model(dataset)
        self.train(model, input_matrix, verbose)
        row_factor = model.row_factors[0].eval()
        col_factor = model.col_factors[0].eval()
        return ALSRecommenderModel(row_factor, col_factor, mapping)
```

O primeiro passo é a transformação de uma lista de ratings em uma matriz esparsa de usuários por vídeos, implementado no método `sparse_input`:

```python
def sparse_input(self, dataset):
    mapping = new_mapping(dataset)

    indices = [(mapping.users_to_idx[r.user_id],
                mapping.items_to_idx[r.item_id])
               for r in dataset.ratings]
    values = [r.rating for r in dataset.ratings]
    shape = (dataset.n_users, dataset.n_items)

    return tf.SparseTensor(indices, values, shape), mapping
```

O segundo passo é a construção do ALS para calcular os fatores e 'completar' os ratings, implementado no método `als_model`:

```python
def als_model(self, dataset):
    return WALSModel(
        dataset.n_users,
        dataset.n_items,
        self.num_factors,
        regularization=self.regularization,
        unobserved_weight=0)
```

O tercero passo é a execução do ALS em si, que consiste na repetição de dois passos: mantem a matriz de vídeos constante e altera a matriz de usuários; mantem a matriz de usuários constante e altera a matriz de vídeos. A cada passo, o erro entre os ratings do input e os ratings aproximados deve diminuir.

Execução do ALS implementada no método `train`:

```python
def train(self, model, input_matrix, verbose=False):
    rmse_op = self.rmse_op(model, input_matrix) if verbose else None

    row_update_op = model.update_row_factors(sp_input=input_matrix)[1]
    col_update_op = model.update_col_factors(sp_input=input_matrix)[1]

    model.initialize_op.run()
    model.worker_init.run()
    for _ in range(self.num_iters):
        # Update Users
        model.row_update_prep_gramian_op.run()
        model.initialize_row_update_op.run()
        row_update_op.run()
        # Update Items
        model.col_update_prep_gramian_op.run()
        model.initialize_col_update_op.run()
        col_update_op.run()

        if verbose:
            print('RMSE: {:,.3f}'.format(rmse_op.eval()))
```

**ALSRecommenderModel**

A classe `ALSRecommenderModel` recebe três parâmetros:

* `user_factors`: matriz densa de usuários por número de fatores
* `item_factors`: matriz densa de vídeos por número de fatores
* `mapping`: objeto que converte `user_id` para / de índice em `user_factors`, `item_id` para / de índice em `item_factors` (vídeos)

A classe `ALSRecommenderModel` implementa dois métodos:

* `transform`: recebe uma lista de `(user_id, item_id)` e retorna a predição do rating
* `recommend`: recebe um `user_id` e retorna a lista de `(item_id, rating)` ordenada com os maiores ratings primeiro

O método `transform` é o produto dos fatores do usuário e do vídeo:

```python
def transform(self, x):
    for user_id, item_id in x:
        if user_id not in self.mapping.users_to_idx \
            or item_id not in self.mapping.items_to_idx:
            yield (user_id, item_id), 0.0
            continue
        i = self.mapping.users_to_idx[user_id]
        j = self.mapping.items_to_idx[item_id]
        u = self.user_factors[i]
        v = self.item_factors[j]
        r = np.dot(u, v)
        yield (user_id, item_id), r
```

O método `recommend` é o produto da matriz de vídeos pelo vetor de fatores de um usuário:

```python
def recommend(self, user_id, num_items=10, items_exclude=set()):
    i = self.mapping.users_to_idx[user_id]
    u = self.user_factors[i]
    V = self.item_factors
    P = np.dot(V, u)
    rank = sorted(enumerate(P), key=lambda p: p[1], reverse=True)

    top = list()
    k = 0
    while k < len(rank) and len(top) < num_items:
        j, r = rank[k]
        k += 1

        item_id = self.mapping.items_from_idx[j]
        if item_id in items_exclude:
            continue

        top.append((item_id, r))

    return top
```

**Execução**

A execução consiste em instanciar a classe `ALSRecommender`, fazer o treinamento com o método `fit` e fazer inferências com a instância da classe `ALSRecommenderModel` retornada.

Nesse exemplo, a inferência é executada para todos os ratings de avaliação como definido na Preparação de Dados. Com o rating da inferência, é calculado o RMSE de cada conjunto de avaliação.

```python
als = ALSRecommender(num_factors=10, num_iters=10, reg=0.1)
print('Training...\n')
als_model = als.fit(train_data, verbose=True)
print('\nEvaluation...\n')
eval_rmse(als_model)
```

Saída:

```html
Training...

RMSE: 1.729
RMSE: 0.765
RMSE: 0.631
RMSE: 0.588
RMSE: 0.565
RMSE: 0.550
RMSE: 0.540
RMSE: 0.532
RMSE: 0.526
RMSE: 0.521

Evaluation...

RMSE (train): 0.521
RMSE (validation): 1.688
RMSE for heavy: 2.444
RMSE for moderate: 1.465
RMSE for accidental: 1.926
```

## Treinamento com Spark

Documentação da implementação:

[ [Manual](http://spark.apache.org/docs/2.1.0/ml-collaborative-filtering.html) ]

Documentação dos parâmetros do ALS:

[ [Python API](http://spark.apache.org/docs/2.1.0/api/python/pyspark.ml.html#module-pyspark.ml.recommendation) ]

Exemplo do ALS:

[ [GitHub: als_example.py](https://github.com/apache/spark/blob/v2.1.0/examples/src/main/python/ml/als_example.py) ]

...

A execução consiste em instanciar a classe `ALS`, fazer o treinamento com o método `fit` e fazer inferências com a instância da classe `ALSModel` retornada.

Nesse exemplo, a inferência é executada para todos os ratings de avaliação como definido na Preparação de Dados. Com o rating da inferência, é calculado o RMSE de cada conjunto de avaliação.

```python
from pyspark.ml.recommendation import ALS as SparkALS

spark_als = SparkALS(rank=10, maxIter=10, regParam=0.1)
spark_model = spark_als.fit(train_df)
eval_rmse_spark(spark_model)
```

Saída:

```html
RMSE (train): 0.601
RMSE (validation): 1.018
RMSE for heavy: 1.128
RMSE for moderate: 0.974
RMSE for accidental: 1.327
```

## Seleção de Parâmetros

A Seleção de Parâmetros consiste em uma busca executando todas as combinações de valores dos parâmetros. Para limitar a busca, é pré-selecionado um conjunto de valores que faz mais sentido.

Nessa análise, foi usada uma seleção de valores ainda menor, visando agilizar o processo.

```python
default_params = dict(num_factors=[5, 10, 20, 50, 100, 200],
                      num_iters=[5, 10, 25],
                      reg = [1e-5, 1e-3, 1e-1, 0.0, 1])

small_params = dict(num_factors=[5, 10, 20],
                    num_iters=[5],
                    reg = [1e-3, 1e-1, 1])

def grid_search(eval_func, params=default_params, verbose=False):
    best_rmse = None
    best_params = None
    for reg in params['reg']:
        for num_iters in params['num_iters']:
            for num_factors in params['num_factors']:
                if verbose:
                    print('\nParams:', num_factors, num_iters, reg)
                try:
                    rmse = eval_func(num_factors, num_iters, reg)
                except:
                    rmse = None
                if verbose:
                    print('RMSE:',
                          '{:,.3f}'.format(rmse) if rmse is not None else '-')
                if rmse is not None and (best_rmse is None or rmse < best_rmse):
                    if verbose:
                        print('best update!')
                    best_rmse = rmse
                    best_params = (num_factors, num_iters, reg)
    return best_params, best_rmse
```

...

**TensorFlow ALS**

```python
def tf_eval(num_factors, num_iters, reg):
    als = ALSRecommender(num_factors=num_factors, num_iters=num_iters, reg=reg)
    model = als.fit(train_data)
    return _rmse(model, valid_eval)

tf_params, tf_score = grid_search(tf_eval, params=small_params, verbose=True)
print()
print('Best Params:\n\nn_factors={}, n_iters={}, reg={}, RMSE={:.3f}' \
        .format(*tf_params, tf_score))
```

Saída:

```html
Params: 5 5 0.001
RMSE: 1.146
best update!

Params: 10 5 0.001
RMSE: 1.309

Params: 20 5 0.001
RMSE: 2.870

Params: 5 5 0.1
RMSE: 1.355

Params: 10 5 0.1
RMSE: 1.438

Params: 20 5 0.1
RMSE: 1.636

Params: 5 5 1
RMSE: 1.487

Params: 10 5 1
RMSE: 1.941

Params: 20 5 1
RMSE: 1.933

Best Params:

n_factors=5, n_iters=5, reg=0.001, RMSE=1.146
```

...

**Spark ALS**

```python
def spark_eval(num_factors, num_iters, reg):
    als = SparkALS(rank=num_factors, maxIter=num_iters, regParam=reg)
    model = als.fit(train_df)
    return _rmse_spark(model, valid_df)

spark_params, spark_score = grid_search(spark_eval, params=small_params, verbose=True)
print()
print('Best Params:\n\nn_factors={}, n_iters={}, reg={}, RMSE={:.3f}' \
        .format(*spark_params, spark_score))
```

Saída:

```html
Params: 5 5 0.001
RMSE: 1.300
best update!

Params: 10 5 0.001
RMSE: 1.418

Params: 20 5 0.001
RMSE: 1.615

Params: 5 5 0.1
RMSE: 0.981
best update!

Params: 10 5 0.1
RMSE: 1.003

Params: 20 5 0.1
RMSE: 1.033

Params: 5 5 1
RMSE: 1.258

Params: 10 5 1
RMSE: 1.258

Params: 20 5 1
RMSE: 1.258

Best Params:

n_factors=5, n_iters=5, reg=0.1, RMSE=0.981
```

## Comparação

Para a comparação das implementações, a medida de performance é o RMSE dos ratings de avaliação do dataset de Teste com os os melhores parâmetros selecionados na busca.

O TensorFlow ALS com 5 fatores, 5 iterações e 0.001 de regularização tem RMSE de 1,183 no Teste.

O Spark ALS com 5 fatores, 5 iterações e 0.1 de regularização tem RMSE de 1,086 no Teste.

Esse resultado mostra que o Spark tem performance melhor que o TensorFlow no erro final.

**TensorFlow ALS**

```python
als = ALSRecommender(*tf_params)
model = als.fit(train_data)
rmse = _rmse(model, test_eval)
print('TensorFlow RMSE for test: {:,.3f}'.format(rmse))
```

Saída:

```html
TensorFlow RMSE for test: 1.183
```

**Spark ALS**

```python
num_factors, num_iters, reg = spark_params
als = SparkALS(rank=num_factors, maxIter=num_iters, regParam=reg)
model = als.fit(train_df)
rmse = _rmse_spark(model, test_df)
print('Spark RMSE for test: {:,.3f}'.format(rmse))
```

Saída:

```html
Spark RMSE for test: 1.086
```

## Conclusão

Apesar do resultado dessa análise ser consistente (várias execuções, pouca variação), não é definitivo. Seria necessário fazer a análise com datasets maiores e verificar se há realmente diferença significativa de performance entre essas implementação.

Para trabalhos futuros, a ideia completar o trabalho com os passos necessários para colocar esse algoritmo 'em produção', ou seja, que um sistema de recomendação possa fazer inferência com o modelo treinado com o ALS do TensorFlow. Uma forma de fazer isso é transformar a classe `ALSRecommenderModel` em um grafo do TensorFlow que possa ser carregado e executado pelo TensorFlow Serving. Esse pode ser o tema de um próximo artigo.

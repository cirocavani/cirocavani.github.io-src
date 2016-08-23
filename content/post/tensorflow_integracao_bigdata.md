+++
date = "2016-08-22T22:00:00-03:00"
draft = false
slug = "tensorflow-integracao-bigdata"
tags = ["TensorFlow", "BigData", "Hadoop"]
title = "TensorFlow: Integração com BigData"
+++

Esse artigo é sobre a criação de uma Aplicação com TensorFlow em que o treinamento é feito no YARN (Hadoop), o servidor de inferência é hospedado no Tsuru e as requisições são feitas por Aplicações Java/Scala. Esses são os desafios para colocar em produção na Globo.com aplicações de Inteligência Artificial. Nesse trabalho foram desenvolvidos projetos que são Provas de Conceito de como fazer essa Aplicação TensorFlow integrada com BigData (o código está disponível no GitHub).

**Código**

https://github.com/cirocavani/tensorflow-poc


## Motivação

O problema começou com: como criar uma Aplicação de AI? A proposta para resolver esse problema foi: TensorFlow.

O problema passou a ser: o que é o TensorFlow? Como criar uma Aplicação com TensorFlow? como colocar essa Aplicação em Produção? como usar essa Aplicação em um produto para entregar valor à empresa e ao usuário?

A preparação para resolver esse problema começou a ser feita no início de 2016, no primeiro Hackday em Janeiro e teve um "evento crucial" no final de Junho, quando o Google publicou em seu blog de pesquisa o uso do TensorFlow para recomendação de app no Google Play.

O TensorFlow é uma biblioteca usada para treinar algoritmos com dados. Esses "algoritmos" podem ser "executados" para gerar resultados que podem ser usados em "aplicações inteligentes". Um exemplo de aplicação que pode usar um "algoritmo" do TensorFlow é Recomendação.

O problema tornou-se: como usar o TensorFlow em Recomendação?

Para resolver esse problema, dividi em duas frentes de trabalho: integrar o TensorFlow na infra de BigData, e; desenvolver uma aplicação de TensorFlow para Recomendação.

A primeira frente de trabalho está feita.


## Por que o TensorFlow?

Nos últimos anos, a tendência que eu venho observando do mercado é que AI finalmente se tornou a área de diferenciação, inovação e crescimento. Os investidores estão cada vez mais colocando dinheiro nessa área. As grandes empresas estão cada vez mais se posicionando como "empresas de AI", sendo os maiores exemplos Google, Facebook e Microsoft, com Apple e Amazon chegando junto. Startups como a que fez a app Prisma valem bilhão de dólares.

Enfim, AI é importante.

A minha expectativa é usar Inteligência Artificial na Globo.com, incluindo produção e distribuição de conteúdo, desenvolvimento de produto, controle de infraestrutura, análises de segurança, ... seguir a tendência de Google, Facebook, Microsoft, Amazon, e também ser uma empresa de AI.

Para o trabalho de fazer AI na Globo.com, a tecnologia escolhida foi o TensorFlow, framework de AI do Google. A estratégia escolhida para acelerar o progresso desse trabalho foi usar a infraestrutura de BigData que já tem bastante poder de processamento. Essa é a proposta inicial, com a expectativa de criar demanda suficiente para uma infraestrutura de AI com GPUs.

O TensorFlow é um framework para construção de algoritmos que podem ser usados para processar texto, imagem e vídeo, para tratamento de linguagem natural, reconhecimento de objetos e faces, reconhecimento de padrões, construção de chatbots, construção de bots para jogos, ..., ou seja, é uma biblioteca de funções avançadas que o Google está usando para fazer AI e que está ganhando muita popularidade entre os desenvolvedores. Contudo, há um risco nesse controle direto do Google sobre uma tecnologia que está se tornando a "base de tudo".

Recentemente, duas reportagens discutiram o TensorFlow.

> **Here's Why Google Is Open-Sourcing Some Of Its Most Important Technology**
>
> http://www.forbes.com/sites/gregsatell/2016/07/18/heres-why-google-is-open-sourcing-some-of-its-most-important-technology/
>
> **Google Sprints Ahead in AI Building Blocks, Leaving Rivals Wary**
>
> http://www.bloomberg.com/news/articles/2016-07-21/google-sprints-ahead-in-ai-building-blocks-leaving-rivals-wary

Portanto, o primeiro passo na direção de "fazer AI na Globo.com" está sendo criar uma primeira Aplicação com TensorFlow.

Esse artigo é sobre como isso foi feito.


### O que é o TensorFlow?

TensorFlow é um framework para processamento de tensores (arrays multidimensionais) em ambientes heterogêneos (CPUs, GPUs, mobile). É um projeto open-source do Google liberado em Novembro/2015. A API principal é Python e a engine de execução é C++, o Google espera que a comunidade desenvolva bindings para outras linguagens.

Na prática, o TensorFlow permite escrever algoritmos na forma de operações com tensores (arrays) que resultam em um grafo de execução. Esse grafo é otimizado e as operações são distribuídas para serem executadas em CPU ou GPU de acordo com o tipo de operação e a disponibilidade desses recursos. A engine pode ser estendida com novas operações que podem ter implementações para CPU e/ou GPU e serão usadas de acordo com os recursos. Essa capacidade de otimizar o processamento para recursos heterogêneos é o principal benefício do TensorFlow.

O Google usa o TensorFlow para pesquisa e desenvolvimento de produtos que usam Deep Learning. Contudo, a proposta é que o framework seja usado para Machine Learning em geral. Essa proposta não torna o TensorFlow um framework para processamento geral, mas essa é uma possibilidade para o futuro.

Atualmente, o Google espera que a comunidade use o TensorFlow para criar algoritmos de Machine Learning que, em um segundo momento, poderão se tornar a biblioteca de algoritmos do TensorFlow (hoje, ainda é escasso comparado aos líderes do mercado, scikit-learn e R/CRAN).

Comparado com Spark que é um framework para processamento geral e que tem uma biblioteca de Machine Learning com vários algoritmos (com suporte a batch, stream, SQL e grafos), acredito que o TensorFlow tem um mecanismo de execução mais "moderno". O processamento otimizado com recursos heterogêneos é um benefício que hoje não está disponível no Spark e pode se tornar indispensável para os processamentos que estão em demanda na atualidade. Ou seja, o TensorFlow parte de um modelo de execução baseado na demanda crescente de algoritmos inteligentes para processar dados enquanto o Spark é a evolução do modelo antigo de paralelização de processamento que pode estar com os dias contados.

O TensorFlow ainda tem muito o que evoluir para que se torne o framework padrão em processamento de dados e Machine Learning, mas me parece que esse é o futuro para o qual estamos caminhando.

Por fim, Jeff Dean, criador do MapReduce e muitas outras tecnologias de processamento distribuído e BigData, é um dos líderes do TensorFlow, o que dá ao projeto muita credibilidade.

Para saber mais sobre o TensorFlow:

> **TensorFlow**<br>
> TensorFlow is an Open Source Software Library for Machine Intelligence
>
> https://www.tensorflow.org/
>
> https://en.wikipedia.org/wiki/TensorFlow
>
> (código)
>
> https://github.com/tensorflow/tensorflow
>
> (blog)
>
> **TensorFlow - Google’s latest machine learning system, open sourced for everyone**
>
>  http://googleresearch.blogspot.com.br/2015/11/tensorflow-googles-latest-machine_9.html
>
> **TensorFlow: Open source machine learning**
>
> https://www.youtube.com/watch?v=oZikw5k_2FM
>
> (Paper)
>
> **TensorFlow: Large-scale machine learning on heterogeneous systems**
>
> http://download.tensorflow.org/paper/whitepaper2015.pdf
>

Os dois últimos releases do TensorFlow foram bastante interessantes:

> **TensorFlow v0.9 now available with improved mobile support**
>
> https://developers.googleblog.com/2016/06/tensorflow-v09-now-available-with.html
>
> Esse é o último release e destaca o uso do TensorFlow em aplicações Mobile, onde é possível treinar um modelo que executa no smartphone para reconhecimento de objeto usando o vídeo da câmera, em tempo real.
>
> **Announcing TensorFlow 0.8 – now with distributed computing support!**
>
> https://research.googleblog.com/2016/04/announcing-tensorflow-08-now-with.html
>
> Nesse release, o destaque foi a funcionalidade de treinamento distribuído do modelo, onde antes só era possível usar uma máquina, agora é possível treinar com várias tanto para distribuir processamento quanto para paralelizar configurações distintas do modelo.


### Experiência com TensorFlow

Já há algum tempo, venho trabalhando esporadicamente com o TensorFlow (nos últimos 3 hackdays na Globo.com, quase 9 meses). Recentemente, o Google divulgou um paper em que eles elaboram como usam o TensorFlow para fazer recomendação de apps no Google Play. Essa se tornou a oportunidade que eu encontrei para começar a trazer para a Globo.com a base para crescermos na área de AI.

> **Wide & Deep Learning: Better Together with TensorFlow**
>
> https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html
>
> **Wide & Deep Learning for Recommender Systems**
>
> http://arxiv.org/abs/1606.07792

De forma prática, o Google disponibilizou no TensorFlow uma API para criar um modelo e um paper que explica como eles usam esse modelo para fazer recomendação. Eles não disponibilizaram o "sistema de recomendação" propriamente dito que, na verdade, precisa ser "construído" a partir do TensorFlow.

Portando, o trabalho consiste em entender a proposta de "como" fazer recomendação usando o TensorFlow, mapear esse "sistema" no "modus operandi" do Ambiente de BigData da Globo.com e avaliar o ganho de ter o TensorFlow como parte dessa plataforma.

Eu comecei a fazer esse trabalho.

Estou trabalhando em duas "frentes": a primeira é genérica sobre integração do TensorFlow em BigData e a outra é específica sobre fazer recomendação com TensorFlow. Estou trabalhando em ambas, mas o foco desse artigo é na primeira porque acredito que seja mais interessante para todos que ainda não conhecem o TensorFlow.


#### Aplicação TensorFlow

A primeira "frente" de trabalho está sendo ganhar experiência em como o **TensorFlow** funciona e identificar como é possível acomodar esse processo na plataforma da Globo.com.

> **TensorFlow - Google’s latest machine learning system, open sourced for everyone**
>
> https://research.googleblog.com/2015/11/tensorflow-googles-latest-machine_9.html

O **TensorFlow** é uma biblioteca C++ com uma API em Python para criação de modelos (especificação do algoritmo / grafo de operações e parâmetros; processamento de dados / treinamento). O resultado do treinamento (e validação) é um "modelo de inferência" que deve ser "transferido" para a "aplicação" que vai "fazer" AI.

Na prática, uma vez definido o grafo de operações e parâmetros, esse "programa" é alimentado com dados que vão ajustando os parâmetros. No final, esse grafo com parâmetros "aprendidos" é transformado em um arquivo que pode ser "carregado" na aplicação e usado para fazer predições / inferência.

Essa é a fase do treinamento e é praticamente toda em Python.

Uma vez que o "modelo de inferência" está feito, a próxima fase é fazer inferência.

Em Recomendação, "inferência" seria fazer uma lista das matérias que mais interessam o Usuário. Esse resultado pode ser obtido de duas formas: off-line e on-line. No primeiro caso, uma aplicação pega todas as matérias disponíveis e executa a inferência em batch guardando os melhores resultados para cada um dos Usuário conhecidos (é isso que fazemos com o algoritmo de Fatoração de Matriz / ALS). No segundo caso, o modelo é usado na requisição na API de Conteúdo para gerar a recomendação para um Usuário (é isso que fazemos com o algoritmo do TF-IDF e gostaríamos de fazer com o ALS também). Considero a primeira solução muito custosa computacionalmente e lenta para refletir o interesse imediato / recente do Usuário. Acredito que a solução on-line é melhor e tem alguns casos de uso em que é a única que faz sentido.

Portanto, a dúvida é: é possível fazer inferência em um modelo do TensorFlow de forma on-line?

No início do ano, o Google tornou público um segundo projeto chamado **TensorFlow Serving** que assume esse papel de executar inferências em modelos do TensorFlow de forma on-line.

> **Running your models in production with TensorFlow Serving**
>
> https://research.googleblog.com/2016/02/running-your-models-in-production-with.html

O **TensorFlow Serving** é uma biblioteca C++ para construção de "servidor" de inferência genérico, que já tem suporte para o protocolo HTTP2 usando gRPC (ProtoBuf) e integração com a biblioteca do TensorFlow para fazer inferência (incluindo atualização automática de modelos).

Na prática, o TF Serving possibilita escrever APIs para modelos do TensorFlow. Uma aplicação C++ que recebe requisições em HTTP2 e executa a inferência ("predição") com o modelo treinado. O principal benefício dessa biblioteca é automaticamente carregar novas versões dos modelos conforme eles vão sendo atualizados pelo treinamento.

Essa é a fase da inferência on-line do modelo e é praticamente toda em C++.


#### Algoritmo TensorFlow

A outra "frente" de trabalho é paralela a primeira e consistem em desenvolver propriamente o algoritmo de recomendação baseado no paper do Google. Isso significa definir as features que serão usadas, como ler esses dados e alimentar o treinamento do modelo e como executar a inferência usando as matérias de um determinado Produto (Portal).

Comecei a fazer esse trabalho também, mas vou deixar para discuti-lo em outro artigo.


## TensorFlow em BigData

O trabalho consistiu em identificar e resolver todos os requisitos para o desenvolvimento de uma Aplicação TensorFlow em uma infraestrutura de BigData.

Para esse trabalho, a "anatomia" de uma Aplicação TensorFlow que foi considerada consiste em dois componentes:

1. o treinamento (aprendizado): o código é em Python, precisa de acesso a dados e poder de processamento.

    Esse programa deve ser empacotado para rodar no YARN (Hadoop, RedHat EL 6)

2. a API de consulta (inferência): o código é em C++, recebe requisições com dados "reais" e retorna o resultado a partir da versão mais recente de um "modelo treinado"

    Esse programa deve ser empacotado para rodar no Tsuru (Ubuntu LTS 14.04)

...

Para integrar uma aplicação que usa essa "Inteligência Artificial", é necessário usar um Cliente que "faça requisições" ao servidor de inferência (2).

Essa funcionalidade pode ser implementada em qualquer linguagem e é feita com Python nos exemplos do TensorFlow.

Para esse trabalho, o interesse é integrar com Aplicações em Scala, logo um cliente Java satisfaz o requisito (Java 7).


## Provas de Conceito

O trabalho consistiu no desenvolvimento de Provas de Conceito (POC) que exploram os desafios para se criar uma Aplicação TensorFlow.

Todas as POCs rodam dentro do Docker e foram testadas no Linux e no Mac (usando Docker on Mac).

Além do Docker, não é necessário mais nada instalado na máquina.

A POC [tflearn_wide_n_deep](#tflearn-wide-n-deep) foi o aquecimento rodando um exemplo do TensorFlow.

As POCs [tfserving_basic](#tfserving-basic), [tfserving_advanced](#tfserving-advanced) e [skeleton_project](#skeleton-project) correspondem ao trabalho de criar um projeto que consiste de algoritmo Python para treinamento e servidor C++ para servir o modelo (tem um cliente Python para validar o funcionamento).

As POCs [yarn_training](#yarn-training), [tensorflow_centos6](#tensorflow-centos6), [tensorflow_installer](#tensorflow-installer), [hadoop_centos6](#hadoop-centos6) e [hadoop_ubuntu1604](#hadoop-ubuntu1604) correspondem ao trabalho de fazer o treinamento usando YARN (Hadoop) no RedHat EL 6 (produção).

A POC [client_java](#client-java) corresponde ao trabalho de usar em uma aplicação Java/Scala um serviço do TensorFlow.

A POC [server_tsuru](#server-tsuru) corresponde ao trabalho de criar uma app no Tsuru para servir um modelo treinado do TensorFlow.


#### tflearn_wide_n_deep

A POC é a execução do tutorial sobre o "algoritmo" usado na recomendação de app do Google Play.

Na prática, é um classificador binário que responde se uma pessoa ganha mais ou menos de 50 mil dólares baseado em dados do Censo, é uma combinação de Logistic Regression com Rede Neural.

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/tflearn_wide_n_deep/README.md)


#### tfserving_basic

A POC é a execução do tutorial sobre o TensorFlow Serving sem versionamento do modelo.

O "algoritmo" desse tutorial é um classificador de imagem que faz reconhecimento de dígito usando dataset MNIST e Rede Neural.

Na prática, consiste de todo o processo de compilação do TensorFlow Serving e do TensorFlow para execução dos três requisitos desse trabalho: treinamento, servidor e cliente (essa "arquitetura" é o resultado final esperado para uma Aplicação TensorFlow).

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/tfserving_basic/README.md)


#### tfserving_advanced

A POC é a execução do tutorial sobre o TensorFlow Serving com versionamento do modelo.

O "algoritmo" desse tutorial é um classificador de imagem que faz reconhecimento de dígito usando dataset MNIST e Rede Neural.

Na prática, consiste de todo o processo de compilação do TensorFlow Serving e do TensorFlow para execução dos três requisitos desse trabalho: treinamento, servidor e cliente (essa "arquitetura" é o resultado final esperado para uma Aplicação TensorFlow).

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/tfserving_advanced/README.md)


#### skeleton_project

(código do exemplo extraído para um projeto fora da árvore do TensorFlow Serving)

A POC é a criação de um projeto standalone baseado no código do tutorial sobre o TensorFlow Serving com versionamento do modelo.

O "algoritmo" desse projeto é um classificador de imagem que faz reconhecimento de dígito usando dataset MNIST e Rede Neural.

Na prática, consiste em separar o código para treinamento, servidor e cliente em um projeto que depende do TensorFlow Serving e usa a mesma ferramenta de build Bazel (fazendo o processo de compilação do TensorFlow Serving e do TensorFlow).

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/skeleton_project/README.md)


#### yarn_training

(motivação: usar a infraestrutura de armazenamento e processamento do Hadoop para rodar o treinamento com TensorFlow)

A POC é a criação de uma Aplicação YARN para rodar o treinamento com TensorFlow.

Na prática, consiste em criar uma aplicação Java baseada no exemplo de execução de shell script distribuído do YARN, essa aplicação é dividida em duas partes, uma que faz submissão do script e a outra que controla dentro do cluster a execução.

Esse procedimento depende do instalador do TensorFlow criado em [tensorflow_installer](#tensorflow-installer) por link simbólico e do container em [hadoop_centos6](#hadoop-centos6) estar rodando.

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/yarn_training/README.md)


#### tensorflow_centos6

(motivação: o TensorFlow não é oficialmente suportado no RedHat EL 6, o binário é compilado para glibc 2.17 e o código C++ 11, ambos requisitos não disponíveis, mas é possível criar um binário do TensorFlow compatível)

A POC é a construção do binário do TensorFlow compatível com RedHat EL 6.

Na prática, consiste em instalar a versão mais recente do GCC disponível para o CentOS 6, construir a ferramenta de build Bazel (binário incompatível com a glibc) e construir o TensorFlow (com patch para linkage).

Essa POC é só para criar um pacote do TensorFlow.

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/tensorflow_centos6/README.md)


#### tensorflow_installer

(motivação: evitar download no ambiente de produção e evitar enviar múltiplos arquivos para o Hadoop)

A POC é a criação de um instalador para o algoritmo de treinamento com o TensorFlow que tenha todas as dependência e possa ser executado no RedHat EL 6.

Na prática, consiste em criar um pacote com TensorFlow, Python (conda) e todas as dependências que é embutido em um shell script que faz a instalação e executa o treinamento do TensorFlow.

Esse procedimento depende do pacote do TensorFlow criado em [tensorflow_centos6](#tensorflow-centos6) por link simbólico.

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/tensorflow_installer/README.md)


#### hadoop_centos6

(motivação: essa imagem corresponde a uma "aproximação" do ambiente de produção que usa RedHat EL 6 com o qual o CentOS6 é binário-compatível - o TensorFlow não é oficialmente suportado nesse sistema)

A POC é a configuração mínima do Hadoop no CentOS 6 para execução do treinamento com TensorFlow no YARN.

Na prática, consiste em rodar os servidores do HDFS (NameNode e DataNode) e do YARN (ResourceManager e NodeManager) para poder executar uma aplicação (ApplicationMaster) que instale o TensorFlow e rode o script Python de treinamento.

Essa POC é só a configuração do Hadoop no CentOS 6.

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/hadoop_centos6/README.md)


#### hadoop_ubuntu1604

(motivação: essa imagem corresponde ao ambiente em que o TensorFlow é oficialmente suportado, diferente do ambiente de produção RedHat, ou seja, o comportamento nesse ambiente deve representar o "funcionamento correto")

A POC é a configuração mínima do Hadoop no Ubuntu para execução do treinamento com TensorFlow no YARN.

Na prática, consiste em rodar os servidores do HDFS (NameNode e DataNode) e do YARN (ResourceManager e NodeManager) para poder executar uma aplicação (ApplicationMaster) que instale o TensorFlow e rode o script Python de treinamento.

Essa POC é só a configuração do Hadoop no Ubuntu 16.04.

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/hadoop_ubuntu1604/README.md)


#### client_java

A POC é a criação de um cliente Java para fazer inferência em um serviço do TensorFlow.

Na prática, consiste em gerar o código do protocolo de comunicação usando o gRPC (Protobuf) e usar esse código para acessar o serviço do TensorFlow, a especificação do protocolo faz parte da implementação do serviço.

O "algoritmo" desse projeto é um classificador de imagem que faz reconhecimento de dígito usando dataset MNIST e Rede Neural.

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/client_java/README.md)


#### server_tsuru

A POC é a criação de uma aplicação do Tsuru para rodar o servidor do TensorFlow.

O "algoritmo" desse servidor é um classificador de imagem que faz reconhecimento de dígito usando dataset MNIST e Rede Neural.

Na prática, consiste em fazer o deploy do binário do servidor do TensorFlow em uma app do Tsuru.

Esse procedimento depende do binário do servidor do TensorFlow criado em [skeleton_project](#skeleton-project).

[README](https://github.com/cirocavani/tensorflow-poc/blob/master/server_tsuru/README.md)


## Conclusão

Esse trabalho é o preparatório para a criação de Aplicações TensorFlow que serão colocadas em Produção na Globo.com.

Ainda tem muitos desafios para completar esse trabalho, mas esse é um bom começo.

O TensorFlow é um projeto fascinante e evolui muito rápido - uma excelente oportunidade de aprendizado e cooperação.

Para trabalhos futuros, esses são alguns dos desafios:

* TFRecord: formato de dados do TensorFlow

    Como treinar um modelo a partir de dados armazenado em Parquet?

* Treinamento Distribuído

    Como treinar um modelo com múltiplos containers no YARN?

* Tensorboard

    Como rodar o Tensorboard no ApplicationMaster para acompanhar o treinamento?

* TF Serving Source para Swift (OpenStack)

    Como armazenar os modelos como objetos no Swift (OpenStack)

...

Em um próximo artigo, pretendo discutir uma Aplicação TensorFlow para Recomendação.

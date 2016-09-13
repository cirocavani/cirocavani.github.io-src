+++
date = "2015-09-11T08:10:00-03:00"
draft = false
slug = "compilacao-do-spark-15-com-bugfix"
tags = ["Spark", "Tutorial"]
title = "Compilação do Spark 1.5 (com bugfix)"
+++

Aproveitando que foi feito o lançamento da versão 1.5.0 do Spark, esse tutorial é sobre a construção do pacote do Spark usando o branch atualizado. O branch foi criado para fazer a estabilização do código que deu origem ao primeiro release. Esse branch continua recebendo atualizações importantes que farão parte de releases bugfix no futuro. Com esse procedimento, é possível gerar o pacote com essas últimas atualizações (e até customizar com alterações próprias) antecipando correções que podem ajudar em produção. Importante entender que ao usar uma versão que não passou pelo release implica em riscos que devem ser mitigados com muitos testes.

Mais informações sobre a construção do Spark podem ser obtidas na documentação [aqui](http://spark.apache.org/docs/latest/building-spark.html).

Mais informações sobre a última versão Spark 1.5.0 no [Release Notes](http://spark.apache.org/releases/spark-release-1-5-0.html) e no blog da Databricks [aqui](https://databricks.com/blog/2015/09/09/announcing-spark-1-5.html).


## Pré-requisito

O procedimento consiste em: provisionar o ambiente; fazer uma cópia do branch estável da última versão, e; gerar o pacote binário e os artefatos do Maven.

As ferramentas necessárias para construção são git, Java 7 e Maven 3.3.

Todo o procedimento é executado na linha de comando do terminal.

(é assumido que o git já está instalado)

**Java**

A versão usada nesse procedimento é o Java 7 para o qual a Oracle já terminou o ciclo de desenvolvimento das releases públicas (gratuitas). Contudo, essa é a versão que tem melhor suporte nas ferramentas que estaremos usando com Spark.

(também tem suporte para o Java 8, mas o interesse é usar esse pacote no Hadoop 2.7 que ainda não suporta oficialmente essa versão)

Segue o procedimento para Linux e MacOSX.

(Linux)

No Linux, para o Java, é usado o JDK da Oracle.

(nesse procedimento, foi usado o ArchLinux atualizado até essa primeira semana de Setembro)

```sh
wget --no-check-certificate --no-cookies --header "Cookie: oraclelicense=accept-securebackup-cookie" http://download.oracle.com/otn-pub/java/jdk/7u80-b15/jdk-7u80-linux-x64.tar.gz

tar zxf jdk-7u80-linux-x64.tar.gz

export JAVA_HOME=`pwd`/jdk1.7.0_80
export PATH=$JAVA_HOME/bin:$PATH

java -version

> java version "1.7.0_80"
> Java(TM) SE Runtime Environment (build 1.7.0_80-b15)
> Java HotSpot(TM) 64-Bit Server VM (build 24.80-b11, mixed mode)
```

(OSX)

No MacOSX, é necessário baixar o pacote no site da Oracle e fazer a instalação.

Download do Java 7 [aqui](http://www.oracle.com/technetwork/java/javase/downloads/java-archive-downloads-javase7-521261.html#jdk-7u80-oth-JPR).

No terminal, a versão específica do Java pode ser configurada ajustando a variável de ambiente:

```sh
export JAVA_HOME="$(/usr/libexec/java_home -v 1.7)"

java -version

> java version "1.7.0_80"
> Java(TM) SE Runtime Environment (build 1.7.0_80-b15)
> Java HotSpot(TM) 64-Bit Server VM (build 24.80-b11, mixed mode)
```

**Maven**

A construção do Spark depende da versão 3.3 do Maven.

```sh
wget http://archive.apache.org/dist/maven/maven-3/3.3.3/binaries/apache-maven-3.3.3-bin.tar.gz

tar zxf apache-maven-3.3.3-bin.tar.gz

export PATH=`pwd`/apache-maven-3.3.3/bin

mvn -version

> Apache Maven 3.3.3 (7994120775791599e205a5524ec3e0dfe41d4a06; 2015-04-22T08:57:37-03:00)
> Maven home: /home/cavani/Software/apache-maven-3.3.3
> Java version: 1.7.0_80, vendor: Oracle Corporation
> Java home: /home/cavani/Software/jdk1.7.0_80/jre
> Default locale: en_US, platform encoding: UTF-8
> OS name: "linux", version: "4.0.4-2-arch", arch: "amd64", family: "unix"
```

## Compilação

Primeiramente é criado um clone local do repositório do Spark no qual é desenvolvido a versão 1.5 (estável).

(use `--depth 1` para baixar apenas os arquivos finais, sem o histórico de mudanças, diminui o download)

```sh
git clone https://github.com/apache/spark.git --branch branch-1.5 spark-1.5

> Cloning into 'spark-1.5'...
> remote: Counting objects: 256928, done.
> remote: Total 256928 (delta 0), reused 0 (delta 0), pack-reused 256928
> Receiving objects: 100% (256928/256928), 121.38 MiB | 1.23 MiB/s, done.
> Resolving deltas: 100% (108225/108225), done.
> Checking connectivity... done.

cd spark-1.5
```

A partir desse branch serão criados todos os releases 1.5.x.

Já foi feito o release da tag v1.5.0 e está aberto o desenvolvimento da versão 1.5.1 (ou seja, a versão corrente no branch é a 1.5.1-SNAPSHOT).

```sh
git log --oneline -30

> 89d351b Revert "[SPARK-6350] [MESOS] Fine-grained mode scheduler respects mesosExecutor.cores"
> (...)
> 2b270a1 Preparing development version 1.5.1-SNAPSHOT
> 908e37b Preparing Spark release v1.5.0-rc3
> 1c752b8 [SPARK-10341] [SQL] fix memory starving in unsafe SMJ
```

O versionamento será com base na versão do último release e a identificação dos bugfix será feita no nome do pacote, preservado a substituição transparente da versão oficial pela atualizada.

```sh
mvn help:evaluate -Dexpression=project.version | grep -v INFO | grep -v WARNING | grep -v Download

> 1.5.1-SNAPSHOT

mvn versions:set -DnewVersion=1.5.0 -DgenerateBackupPoms=false

> (...)
> [INFO] Reactor Summary:
> [INFO]
> [INFO] Spark Project Parent POM ........................... SUCCESS [  4.559 s]
> (...)
> [INFO] BUILD SUCCESS
> (...)
```

Por fim, a construção do pacote.

Nesse caso estaremos construindo um pacote com suporte ao YARN no Hadoop 2.7.1, suporte Hive com JDBC.

Estamos colocando no nome do pacote o número do último commit.

```sh
export MAVEN_OPTS="-Xmx2g -XX:MaxPermSize=512M -XX:ReservedCodeCacheSize=512m"

./make-distribution.sh --name 89d351b --tgz --skip-java-test -Phadoop-2.6 -Pyarn -Phive -Phive-thriftserver -Dhadoop.version=2.7.1

> (...)
> [INFO] Reactor Summary:
> [INFO]
> [INFO] Spark Project Parent POM ........................... SUCCESS [  3.841 s]
> [INFO] Spark Project Launcher ............................. SUCCESS [ 12.819 s]
> [INFO] Spark Project Networking ........................... SUCCESS [ 10.980 s]
> [INFO] Spark Project Shuffle Streaming Service ............ SUCCESS [  6.876 s]
> [INFO] Spark Project Unsafe ............................... SUCCESS [ 15.828 s]
> [INFO] Spark Project Core ................................. SUCCESS [03:19 min]
> [INFO] Spark Project Bagel ................................ SUCCESS [  7.048 s]
> [INFO] Spark Project GraphX ............................... SUCCESS [ 18.493 s]
> [INFO] Spark Project Streaming ............................ SUCCESS [ 41.120 s]
> [INFO] Spark Project Catalyst ............................. SUCCESS [01:01 min]
> [INFO] Spark Project SQL .................................. SUCCESS [01:22 min]
> [INFO] Spark Project ML Library ........................... SUCCESS [01:13 min]
> [INFO] Spark Project Tools ................................ SUCCESS [  2.460 s]
> [INFO] Spark Project Hive ................................. SUCCESS [ 58.477 s]
> [INFO] Spark Project REPL ................................. SUCCESS [ 11.646 s]
> [INFO] Spark Project YARN ................................. SUCCESS [ 14.443 s]
> [INFO] Spark Project Hive Thrift Server ................... SUCCESS [ 11.609 s]
> [INFO] Spark Project Assembly ............................. SUCCESS [02:02 min]
> [INFO] Spark Project External Twitter ..................... SUCCESS [  8.653 s]
> [INFO] Spark Project External Flume Sink .................. SUCCESS [  5.997 s]
> [INFO] Spark Project External Flume ....................... SUCCESS [ 12.408 s]
> [INFO] Spark Project External Flume Assembly .............. SUCCESS [  3.959 s]
> [INFO] Spark Project External MQTT ........................ SUCCESS [ 22.884 s]
> [INFO] Spark Project External MQTT Assembly ............... SUCCESS [  8.830 s]
> [INFO] Spark Project External ZeroMQ ...................... SUCCESS [  8.407 s]
> [INFO] Spark Project External Kafka ....................... SUCCESS [ 14.933 s]
> [INFO] Spark Project Examples ............................. SUCCESS [01:52 min]
> [INFO] Spark Project External Kafka Assembly .............. SUCCESS [  7.171 s]
> [INFO] Spark Project YARN Shuffle Service ................. SUCCESS [  7.010 s]
> [INFO] ------------------------------------------------------------------------
> [INFO] BUILD SUCCESS
> [INFO] ------------------------------------------------------------------------
> [INFO] Total time: 16:08 min
> [INFO] Finished at: 2015-09-11T06:44:55-03:00
> [INFO] Final Memory: 417M/1553M
> [INFO] ------------------------------------------------------------------------
> (...)
```

Resultado:

`spark-1.5.0-bin-89d351b.tgz`

(Artefatos do Maven)

```sh
rm -rf ~/.m2/repository/org/apache/spark

mvn install -Phadoop-2.6 -Pyarn -Phive -Phive-thriftserver -Dhadoop.version=2.7.1 -DskipTests

> (...)
> [INFO] Reactor Summary:
> [INFO]
> [INFO] Spark Project Parent POM ........................... SUCCESS [  4.339 s]
> [INFO] Spark Project Launcher ............................. SUCCESS [ 14.078 s]
> [INFO] Spark Project Networking ........................... SUCCESS [  8.555 s]
> [INFO] Spark Project Shuffle Streaming Service ............ SUCCESS [  3.540 s]
> [INFO] Spark Project Unsafe ............................... SUCCESS [  3.395 s]
> [INFO] Spark Project Core ................................. SUCCESS [01:22 min]
> [INFO] Spark Project Bagel ................................ SUCCESS [  7.293 s]
> [INFO] Spark Project GraphX ............................... SUCCESS [ 15.367 s]
> [INFO] Spark Project Streaming ............................ SUCCESS [ 26.005 s]
> [INFO] Spark Project Catalyst ............................. SUCCESS [ 49.232 s]
> [INFO] Spark Project SQL .................................. SUCCESS [ 48.866 s]
> [INFO] Spark Project ML Library ........................... SUCCESS [01:01 min]
> [INFO] Spark Project Tools ................................ SUCCESS [  8.979 s]
> [INFO] Spark Project Hive ................................. SUCCESS [ 29.601 s]
> [INFO] Spark Project REPL ................................. SUCCESS [ 19.661 s]
> [INFO] Spark Project YARN ................................. SUCCESS [ 16.976 s]
> [INFO] Spark Project Hive Thrift Server ................... SUCCESS [ 13.583 s]
> [INFO] Spark Project Assembly ............................. SUCCESS [02:01 min]
> [INFO] Spark Project External Twitter ..................... SUCCESS [  9.734 s]
> [INFO] Spark Project External Flume Sink .................. SUCCESS [ 10.291 s]
> [INFO] Spark Project External Flume ....................... SUCCESS [ 12.282 s]
> [INFO] Spark Project External Flume Assembly .............. SUCCESS [  4.252 s]
> [INFO] Spark Project External MQTT ........................ SUCCESS [ 21.910 s]
> [INFO] Spark Project External MQTT Assembly ............... SUCCESS [  8.383 s]
> [INFO] Spark Project External ZeroMQ ...................... SUCCESS [  7.677 s]
> [INFO] Spark Project External Kafka ....................... SUCCESS [ 13.317 s]
> [INFO] Spark Project Examples ............................. SUCCESS [01:45 min]
> [INFO] Spark Project External Kafka Assembly .............. SUCCESS [  6.813 s]
> [INFO] Spark Project YARN Shuffle Service ................. SUCCESS [  7.544 s]
> [INFO] ------------------------------------------------------------------------
> [INFO] BUILD SUCCESS
> [INFO] ------------------------------------------------------------------------
> [INFO] Total time: 12:24 min
> [INFO] Finished at: 2015-09-11T07:52:35-03:00
> [INFO] Final Memory: 102M/1349M
> [INFO] ------------------------------------------------------------------------

cd ~/.m2/repository
tar cf spark-1.5.0-m2-89d351b.tar org/apache/spark

cd -
mv ~/.m2/repository/spark-1.5.0-m2-89d351b.tar .
```

Resultado:

`spark-1.5.0-m2-89d351b.tar`


## Conclusão

O Spark é um framework que vem evoluindo rapidamente, com contribuições das mais diversas origem. Praticamente todos os grandes de BigData estão contribuindo com o Spark. Muitas vezes, surgem novas funcionalidades que podem agregar muito valor nas suas aplicações. Outras vezes, são bugs corrigidos que contribuem para a estabilidade de uma aplicação que já existe. Também tem o 'prazer' de ser um 'early adopter'. Seja qual for o motivo, esse procedimento mostra que o trabalho para ter um pacote do Spark é bem fácil e, por experiência, esse é um fator bastante relevante para ganhar tempo e gerar máximo valor.

Nos próximos artigos, vou falar mais de como usar o Spark para desenvolver um aplicação que roda no Hadoop.

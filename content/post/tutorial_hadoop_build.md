+++
date = "2015-08-31T22:45:23-03:00"
draft = false
slug = "compilacao-do-hadoop-para-centos6-rhel6-usando-docker"
tags = ["Tutorial", "Hadoop", "Docker", "CentOS6", "RHEL6"]
title = "Compilação do Hadoop para CentOS6 / RHEL6 usando Docker"
+++

Esse tutorial é sobre a construção do pacote do Hadoop 2.7.1 para o CentOS6 / RHEL6 usando Docker. Esse procedimento é necessário para gerar as bibliotecas nativas compatíveis. O principal objetivo que motivou esse trabalho foi configurar o FairScheduler do YARN usando CGroups rodando no Red Hat Enterprise Linux 6 (RHEL6). O pacote Hadoop distribuído pela Apache tem executável binário que não é compatível com a Glibc que faz parte do CentOS6/RHEL6.

O RHEL6 é o sistema operacional homologado para as máquinas do cluster que usamos na Globo.com e foi necessário criar uma distribuição própria do Hadoop para que pudéssemos fazer uso do [FairScheduler](http://hadoop.apache.org/docs/r2.7.1/hadoop-yarn/hadoop-yarn-site/FairScheduler.html) juntamente com o [CGroups](http://hadoop.apache.org/docs/r2.7.1/hadoop-yarn/hadoop-yarn-site/NodeManagerCgroups.html) para limitar o uso de processamento entre as aplicações rodando nos mesmos NodeManagers.

Esse trabalho de configuração do Hadoop para uso compartilhado será assunto de outro artigo.

Nesse artigo, o foco é um passo a passo de como usar o Docker para gerar um pacote do Hadoop adaptado para o Red Hat Enterprise Linux 6 (RHEL6) usando CentOS6.


## Pré-requisito

Nesse procedimento, é necessário que o Docker esteja instalado e funcionando; também é necessário acesso à Internet.

Originalmente, esse procedimento foi testado no ArchLinux atualizado até final de Agosto/2015.

https://wiki.archlinux.org/index.php/Docker

```sh
sudo docker version

> Client:
>  Version:      1.8.1
>  API version:  1.20
>  Go version:   go1.4.2
>  Git commit:   d12ea79
>  Built:        Sat Aug 15 17:29:10 UTC 2015
>  OS/Arch:      linux/amd64
>
> Server:
>  Version:      1.8.1
>  API version:  1.20
>  Go version:   go1.4.2
>  Git commit:   d12ea79
>  Built:        Sat Aug 15 17:29:10 UTC 2015
>  OS/Arch:      linux/amd64
```


## Compilação

Documento com instruções de build do Hadoop [aqui](https://github.com/apache/hadoop/blob/release-2.7.1/BUILDING.txt).

O resultado desse procedimento é um pacote do Hadoop com os executáveis e bibliotecas nativas compilados para o CentOS6 que rodam no RHEL6.

`/hadoop/hadoop-2.7.1-src/hadoop-dist/target/hadoop-2.7.1.tar.gz`

...

Começamos com a criação de um conainer do Docker com a imagem do CentOS6.

Ao executar o comando `run`, o Docker automaticamente fará o download da imagem e a shell será inicializada dentro de um novo container.

```sh
sudo docker run -i -t centos:6 /bin/bash

> Unable to find image 'centos:6' locally
> 6: Pulling from library/centos
>
> f1b10cd84249: Pull complete
> fb9cc58bde0c: Pull complete
> a005304e4e74: Already exists
> library/centos:6: The image you are pulling has been verified. Important: image verification is a tech preview feature and should not be relied on to provide security.
>
> Digest: sha256:25d94c55b37cb7a33ad706d5f440e36376fec20f59e57d16fe02c64698b531c1
> Status: Downloaded newer image for centos:6
> [root@3cc2bc5e593b /]#
```

Já dentro do container criamos um usuário e local que serão usados na compilação e geração do pacote.

```sh
adduser -m -d /hadoop hadoop
cd hadoop
```

Para a compilação das bibliotecas nativas é necessária a instalação do compilador C e mais alguns pacotes de desenvolvimento (cabeçalhos das bibliotecas usadas pelo Hadoop).

```sh
yum install -y tar gzip gcc-c++ cmake zlib zlib-devel openssl openssl-devel fuse fuse-devel bzip2 bzip2-devel snappy snappy-devel

> (...)
```

O Hadoop ainda depende de duas outras bibliotecas que precisam ser instaladas manualmente no CentOS: Google ProtoBuf 2.5 (RPC), Jansson (JSON).

Para instalar o ProtoBuf, é necessário baixar o pacote, configurar para as pastas do CentOS (64 bits) e instalar.

```sh
curl -L -O https://github.com/google/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.gz
tar zxf protobuf-2.5.0.tar.gz
cd protobuf-2.5.0
./configure --prefix=/usr --libdir=/usr/lib64
make
make check
make install

cd ..
```

Para instalar o Jansson, é necessário baixar o pacote, configurar para as pastas do CentOS (64 bits) e instalar.

```sh
curl -O http://www.digip.org/jansson/releases/jansson-2.7.tar.gz
tar zxf jansson-2.7.tar.gz
cd jansson-2.7
./configure --prefix=/usr --libdir=/usr/lib64
make
make install

cd ..
```

Para completar o ambiente de compilação, precisamos do JDK e do Maven.

No caso do JDK, usaremos o pacote RPM já disponibilizado pela Oracle.

```sh
curl -k -L -H "Cookie: oraclelicense=accept-securebackup-cookie" -O http://download.oracle.com/otn-pub/java/jdk/8u60-b27/jdk-8u60-linux-x64.rpm
rpm -i jdk-8u60-linux-x64.rpm
```

No caso do Maven, usaremos o pacote binário de distribuição da Apache.

```sh
curl -O http://archive.apache.org/dist/maven/maven-3/3.3.3/binaries/apache-maven-3.3.3-bin.tar.gz
tar zxf apache-maven-3.3.3-bin.tar.gz
```

O ambiente  de compilação está completo.

Agora estamos pronto para a compilação do Hadoop. Nesse caso, estaremos gerando o pacote de distribuição somente com o binário Java e as bibliotecas nativas.

```sh
su - hadoop

export PATH=$PATH:/hadoop/apache-maven-3.3.3/bin

curl -O http://archive.apache.org/dist/hadoop/common/hadoop-2.7.1/hadoop-2.7.1-src.tar.gz
tar zxf hadoop-2.7.1-src.tar.gz
cd hadoop-2.7.1-src

mvn clean package -Pdist,native -DskipTests -Drequire.snappy -Drequire.openssl -Dtar

> (...)
> main:
>      [exec] $ tar cf hadoop-2.7.1.tar hadoop-2.7.1
>      [exec] $ gzip -f hadoop-2.7.1.tar
>      [exec]
>      [exec] Hadoop dist tar available at: /hadoop/hadoop-2.7.1-src/hadoop-dist/target/hadoop-2.7.1.tar.gz
>      [exec]
> [INFO] Executed tasks
> [INFO]
> [INFO] --- maven-javadoc-plugin:2.8.1:jar (module-javadocs) @ hadoop-dist ---
> [INFO] Building jar: /hadoop/hadoop-2.7.1-src/hadoop-dist/target/hadoop-dist-2.7.1-javadoc.jar
> [INFO] ------------------------------------------------------------------------
> [INFO] Reactor Summary:
> [INFO]
> [INFO] Apache Hadoop Main ................................. SUCCESS [01:56 min]
> [INFO] Apache Hadoop Project POM .......................... SUCCESS [ 42.134 s]
> [INFO] Apache Hadoop Annotations .......................... SUCCESS [ 37.761 s]
> [INFO] Apache Hadoop Assemblies ........................... SUCCESS [  0.125 s]
> [INFO] Apache Hadoop Project Dist POM ..................... SUCCESS [ 23.183 s]
> [INFO] Apache Hadoop Maven Plugins ........................ SUCCESS [ 25.962 s]
> [INFO] Apache Hadoop MiniKDC .............................. SUCCESS [03:23 min]
> [INFO] Apache Hadoop Auth ................................. SUCCESS [02:11 min]
> [INFO] Apache Hadoop Auth Examples ........................ SUCCESS [ 10.145 s]
> [INFO] Apache Hadoop Common ............................... SUCCESS [03:29 min]
> [INFO] Apache Hadoop NFS .................................. SUCCESS [  4.724 s]
> [INFO] Apache Hadoop KMS .................................. SUCCESS [02:35 min]
> [INFO] Apache Hadoop Common Project ....................... SUCCESS [  0.024 s]
> [INFO] Apache Hadoop HDFS ................................. SUCCESS [02:15 min]
> [INFO] Apache Hadoop HttpFS ............................... SUCCESS [02:13 min]
> [INFO] Apache Hadoop HDFS BookKeeper Journal .............. SUCCESS [ 38.598 s]
> [INFO] Apache Hadoop HDFS-NFS ............................. SUCCESS [  3.213 s]
> [INFO] Apache Hadoop HDFS Project ......................... SUCCESS [  0.032 s]
> [INFO] hadoop-yarn ........................................ SUCCESS [  0.030 s]
> [INFO] hadoop-yarn-api .................................... SUCCESS [ 29.193 s]
> [INFO] hadoop-yarn-common ................................. SUCCESS [02:02 min]
> [INFO] hadoop-yarn-server ................................. SUCCESS [  0.040 s]
> [INFO] hadoop-yarn-server-common .......................... SUCCESS [  8.499 s]
> [INFO] hadoop-yarn-server-nodemanager ..................... SUCCESS [ 12.283 s]
> [INFO] hadoop-yarn-server-web-proxy ....................... SUCCESS [  2.359 s]
> [INFO] hadoop-yarn-server-applicationhistoryservice ....... SUCCESS [  5.298 s]
> [INFO] hadoop-yarn-server-resourcemanager ................. SUCCESS [ 15.095 s]
> [INFO] hadoop-yarn-server-tests ........................... SUCCESS [  3.772 s]
> [INFO] hadoop-yarn-client ................................. SUCCESS [  4.641 s]
> [INFO] hadoop-yarn-server-sharedcachemanager .............. SUCCESS [  2.433 s]
> [INFO] hadoop-yarn-applications ........................... SUCCESS [  0.019 s]
> [INFO] hadoop-yarn-applications-distributedshell .......... SUCCESS [  1.884 s]
> [INFO] hadoop-yarn-applications-unmanaged-am-launcher ..... SUCCESS [  1.263 s]
> [INFO] hadoop-yarn-site ................................... SUCCESS [  0.020 s]
> [INFO] hadoop-yarn-registry ............................... SUCCESS [  3.532 s]
> [INFO] hadoop-yarn-project ................................ SUCCESS [  3.452 s]
> [INFO] hadoop-mapreduce-client ............................ SUCCESS [  0.036 s]
> [INFO] hadoop-mapreduce-client-core ....................... SUCCESS [ 15.195 s]
> [INFO] hadoop-mapreduce-client-common ..................... SUCCESS [ 12.459 s]
> [INFO] hadoop-mapreduce-client-shuffle .................... SUCCESS [  2.645 s]
> [INFO] hadoop-mapreduce-client-app ........................ SUCCESS [  6.342 s]
> [INFO] hadoop-mapreduce-client-hs ......................... SUCCESS [  3.845 s]
> [INFO] hadoop-mapreduce-client-jobclient .................. SUCCESS [ 11.295 s]
> [INFO] hadoop-mapreduce-client-hs-plugins ................. SUCCESS [  1.546 s]
> [INFO] Apache Hadoop MapReduce Examples ................... SUCCESS [  4.573 s]
> [INFO] hadoop-mapreduce ................................... SUCCESS [  2.164 s]
> [INFO] Apache Hadoop MapReduce Streaming .................. SUCCESS [  7.874 s]
> [INFO] Apache Hadoop Distributed Copy ..................... SUCCESS [ 19.660 s]
> [INFO] Apache Hadoop Archives ............................. SUCCESS [  2.071 s]
> [INFO] Apache Hadoop Rumen ................................ SUCCESS [  3.966 s]
> [INFO] Apache Hadoop Gridmix .............................. SUCCESS [  3.215 s]
> [INFO] Apache Hadoop Data Join ............................ SUCCESS [  1.818 s]
> [INFO] Apache Hadoop Ant Tasks ............................ SUCCESS [  1.478 s]
> [INFO] Apache Hadoop Extras ............................... SUCCESS [  2.037 s]
> [INFO] Apache Hadoop Pipes ................................ SUCCESS [  5.880 s]
> [INFO] Apache Hadoop OpenStack support .................... SUCCESS [  3.407 s]
> [INFO] Apache Hadoop Amazon Web Services support .......... SUCCESS [ 40.013 s]
> [INFO] Apache Hadoop Azure support ........................ SUCCESS [ 11.557 s]
> [INFO] Apache Hadoop Client ............................... SUCCESS [  7.659 s]
> [INFO] Apache Hadoop Mini-Cluster ......................... SUCCESS [  0.042 s]
> [INFO] Apache Hadoop Scheduler Load Simulator ............. SUCCESS [  3.072 s]
> [INFO] Apache Hadoop Tools Dist ........................... SUCCESS [  8.519 s]
> [INFO] Apache Hadoop Tools ................................ SUCCESS [  0.014 s]
> [INFO] Apache Hadoop Distribution ......................... SUCCESS [ 30.616 s]
> [INFO] ------------------------------------------------------------------------
> [INFO] BUILD SUCCESS
> [INFO] ------------------------------------------------------------------------
> [INFO] Total time: 29:26 min
> [INFO] Finished at: 2015-09-01T00:47:31+00:00
> [INFO] Final Memory: 224M/785M
> [INFO] ------------------------------------------------------------------------
```

Para completar a compilação, executamos os testes, contudo, alguns deles podem apresentar falhas intermitentes (acontecem algumas vezes, outras não).

Os testes podem levar algumas horas para rodar por completo.

```sh
mkdir hadoop-common-project/hadoop-common/target/test-classes/webapps/test

mvn test -Pnative -Drequire.snappy -Drequire.openssl -Dmaven.test.failure.ignore=true -Dsurefire.rerunFailingTestsCount=3

> (...)
```

(alguns testes com falha intermitente)

* org.apache.hadoop.ipc.TestDecayRpcScheduler#testAccumulate
* org.apache.hadoop.ipc.TestDecayRpcScheduler#testPriority
* org.apache.hadoop.hdfs.server.datanode.TestDataNodeMetrics#testDataNodeTimeSpend
* org.apache.hadoop.hdfs.shortcircuit.TestShortCircuitCache#testDataXceiverHandlesRequestShortCircuitShmFailure

...

No final desse procedimento, o pacote do Hadoop estará gerado em:

`/hadoop/hadoop-2.7.1-src/hadoop-dist/target/hadoop-2.7.1.tar.gz`

Para copiar do container para a máquina host:
<br/>(`3cc2bc5e593b` é o identificador do container no Docker)

```sh
# shell na máquina
sudo docker cp 3cc2bc5e593b:/hadoop/hadoop-2.7.1-src/hadoop-dist/target/hadoop-2.7.1.tar.gz .
```

## Conclusão

Esse procedimento mostra como o Hadoop pode ser customizado para necessidades específicas e que não requer um esforço muito grande.

Contudo, ter uma "versão" própria do Hadoop é uma decisão que deve ser tomada com cautela.

No momento, a gente considera que essa seja a melhor escolha para o nosso trabalho na Globo.com e estamos querendo formar um time para evoluir e dar suporte a essa plataforma. O maior benefício é a liberdade de escolher como configurar e melhorar nossa infraestrutura. O custo é não ter uma empresa especializada "cuidando" dessa responsabilidade.

No futuro, pode ser que mudemos esse modo de operação e busquemos uma distribuição "profissional" como Cloudera, Hortonworks ou outra.

Particularmente, eu prefiro manter uma plataforma própria.

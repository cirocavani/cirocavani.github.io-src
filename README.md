# Personal Homepage

Clonar o projeto:

    git clone --recurse-submodules git@github.com:cirocavani/cirocavani.github.io-src.git cirocavani.github.io

Deploy:

```sh
./deploy.sh
```

Exemplo:

```sh
hugo new post/<NOME_DO_ARQUIVO>.md
hugo server --buildDrafts --watch
#(edita o conteúdo em content/post/<NOME_DO_ARQUIVO>.md)
#(visualiza o resultado em  http://127.0.0.1:1313/)
#(remove a configuração de draft)
./deploy.sh 'Novo artigo sobre ...'
```

Tema:

```sh
cd themes/blackburn/
git remote add upstream https://github.com/yoshiharuyamashita/blackburn.git
git fetch upstream
git rebase upstream/master
git push
```

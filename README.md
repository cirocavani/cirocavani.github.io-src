# Personal Homepage

Clonar o projeto:

    git clone --recurse-submodules git@github.com:cirocavani/cirocavani.github.io-src.git cirocavani.github.io

Deploy:

    ./deploy.sh

Exemplo:

    hugo new post/<NOME_DO_ARQUIVO>.md
    hugo server --buildDrafts --watch
    (edita o conteúdo em content/post/<NOME_DO_ARQUIVO>.md)
    (visualiza o resultado em  http://127.0.0.1:1313/)
    (remove a configuração de draft)
    ./deploy.sh 'Novo artigo sobre ...'

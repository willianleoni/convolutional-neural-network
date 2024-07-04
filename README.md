# Projeto de Rede Neural Convolucional para Classificação de Imagens

Este projeto é um exemplo de implementação de um sistema de Rede Neural Convolucional (CNN) para classificação de imagens de diferentes tipos de componentes, podendo ser adaptado para diferentes conjuntos de dados. A estrutura do projeto está organizada da seguinte maneira:

## Estrutura de Pastas

- **data**: Contém as pastas `train` e `test` com as imagens de treinamento e teste, respectivamente.
- **main**: Contém os principais scripts do projeto.
- **models**: Armazena os modelos treinados.
- **notebooks**: Contém arquivos `.ipynb` usados como modelo para criação e organização da estrutura do projeto.
- **validation**: Contém as imagens geradas e salvas durante a validação do modelo.

## Arquivos Principais

- `main/main.py`: Ponto de entrada do projeto. Este script carrega as configurações do arquivo JSON, cria os dataloaders, define o modelo, otimizador e função de perda, executa o treinamento e validação, e plota as perdas.
- `main/train.py`: Contém a função `train` para o loop de treinamento do modelo.
- `main/validate.py`: Contém a função `validate` para o loop de validação do modelo.

## Arquitetura da CNN

A arquitetura da CNN está definida no arquivo `models/model.py`. A arquitetura inclui camadas de convolução, camadas de pooling e camadas totalmente conectadas. A saída final é mapeada para o número de classes (tipos de componentes de roda [3]).

## Carregamento de Dados

A pasta `data` contém subpastas `train` e `test` com imagens categorizadas. O dataloader carrega as imagens, aplica transformações como redimensionamento, normalização e aumento de dados (se configurado no arquivo `.json`), e divide-os em lotes para treinamento e validação.

## Configurações

As configurações de treinamento estão definidas no arquivo `config.json`. Isso inclui caminhos para salvar as imagens da matriz de confusão e perdas, número de épocas, taxa de aprendizado, se o aumento de dados deve ser aplicado, tamanho do lote e caminhos para os conjuntos de dados.

## Funcionalidades Adicionais

- Funções auxiliares em `utils` para plotar gráficos de perda, calcular a acurácia e criar a matriz de confusão.
- Matriz de confusão e gráficos de perda são gerados durante a validação.
- Hiperparâmetros e resultados são salvos em um dicionário.

O projeto carrega as configurações de um arquivo JSON, cria dataloaders com base nessas configurações, define o modelo da CNN, otimizador e função de perda, executa o treinamento e validação, e fornece análises visuais das métricas de desempenho, salvando os resultados nas respectivas pastas.

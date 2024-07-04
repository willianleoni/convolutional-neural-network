# Projeto de Rede Neural Convolucional para Classificação de Imagens

Este projeto implementa um sistema de Rede Neural Convolucional (CNN) para a classificação de imagens de diferentes tipos de componentes. A estrutura do projeto é modular e pode ser adaptada para diferentes conjuntos de dados.

## Estrutura do Projeto

```plaintext
|-- data/
|   |-- train/
|   |-- test/
|-- main/
|   |-- main.py
|   |-- train.py
|   |-- validate.py
|   |-- data_loader.py
|   |-- model.py
|   |-- utils.py
|-- models/
|   |-- model.pyh
|-- notebooks/
|   |-- *.ipynb
|-- validation/
|   |-- confusion_matrix/
|   |-- loss_plots/
|-- config.json
|-- README.md
```

### Descrição das Pastas

- **data/**: Contém as subpastas `train/` e `test/` com as imagens de treinamento e teste, respectivamente.
- **main/**: Contém os scripts principais do projeto.
  - `main.py`: Ponto de entrada do projeto.
  - `train.py`: Função `train` para o loop de treinamento do modelo.
  - `validate.py`: Função `validate` para o loop de validação do modelo.
- **models/**: Armazena os modelos treinados.
  - `model.py`: Define a arquitetura da CNN.
- **notebooks/**: Contém arquivos `.ipynb` usados para criação e organização da estrutura do projeto.
- **validation/**: Armazena as imagens geradas e salvas durante a validação do modelo.
  - `confusion_matrix/`: Contém as matrizes de confusão geradas.
  - `loss_plots/`: Contém os gráficos de perda gerados.
- **config.json**: Arquivo de configuração para o treinamento do modelo.
- **README.md**: Descrição detalhada do projeto, instruções de uso e detalhes técnicos.

## Arquitetura da CNN

A arquitetura da CNN está definida no arquivo `models/model.py`. Inclui camadas de convolução, camadas de pooling e camadas totalmente conectadas. A saída final é mapeada para o número de classes (tipos de componentes de roda).

## Carregamento de Dados

A pasta `data/` contém subpastas `train/` e `test/` com imagens categorizadas. O `dataloader` carrega as imagens, aplica transformações como redimensionamento, normalização e aumento de dados (se configurado no arquivo JSON), e divide-os em lotes para treinamento e validação.

## Configurações

As configurações de treinamento estão definidas no arquivo `config.json`. Inclui:
- Caminhos para salvar as imagens da matriz de confusão e perdas.
- Número de épocas.
- Taxa de aprendizado.
- Se o aumento de dados deve ser aplicado.
- Tamanho do lote.
- Caminhos para os conjuntos de dados.

## Funcionalidades Adicionais

- Funções auxiliares em `utils` para plotar gráficos de perda, calcular a acurácia e criar a matriz de confusão.
- Matrizes de confusão e gráficos de perda são gerados durante a validação.
- Hiperparâmetros e resultados são salvos em um dicionário.


## Exemplo de Configuração (`config.json`)

```json
{
  "data_path": "data/",
  "train_path": "data/train/",
  "test_path": "data/test/",
  "output_path": "models/",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001,
  "data_augmentation": true,
  "save_plots_path": "validation/loss_plots/",
  "save_confusion_matrix_path": "validation/confusion_matrix/"
}
```
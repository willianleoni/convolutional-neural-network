config.json 

  
1 - Paths: 

        "confusion_matrix": Caminho para salvar a matriz de confusão resultante da avaliação do modelo. Para cada novo treinamento, deve se observar o nome que será utilizado para salvar o arquivo, caso não seja setado corretamente, ele sempre vai sobrescrever o ultimo arquivo gerado com um do mesmo nome. 

        "loss_plot": Caminho para salvar o gráfico das perdas de treinamento e validação, necessário observar as mesmas condições descritas para o confusion_matrix. 

  

2 - Training (Treinamento): 

        "num_epochs": Número de épocas (iterações completas pelos dados de treinamento) para treinar o modelo, pode ser ajustado de acordo com o dataset. 

        "learning_rate": Taxa de aprendizado do otimizador usado para ajustar os pesos do modelo. 

        "data_augmentation": Um valor booleano que determina se a técnica de data augmentation deve ser aplicada durante o treinamento, para mais informações sobre essa técnica, leia o arquivo READ.md dentro da pasta validation. 

        "batch_size": Tamanho do lote de dados usado durante o treinamento, caso o treinamento acuse falta de memória, deve-se diminuir o tamanho do lote de acordo com a memória disponível na CPU/GPU. 

        "train_dataset_path": Caminho para o conjunto de dados de treinamento. 

        "test_dataset_path": Caminho para o conjunto de dados de teste. 

  

Este arquivo de configuração define os caminhos para salvar resultados importantes. Ele também especifica as configurações essenciais para o treinamento, incluindo o número de épocas, taxa de aprendizado, uso de data augmentation, tamanho do lote e caminhos para os conjuntos de dados de treinamento e teste. 

   

Observações importantes: 

	Caso este modelo seja usado para treinar outro dataset, e de extrema importância que se façam as configurações necessárias para as categorias a serem previstas, e organizar as pastas e subpastas de maneira correta. 

	 

	Exemplo de como deve ser feito: 

	 

	   ├── data 

	   ├── train 

	   │   ├── aro 

	   │   │   ├── image1.jpg 

	   │   │   ├── image2.jpg 

	   │   │   └── ... 

	   │   ├── cubo 

	   │   │   ├── image1.jpg 

	   │   │   ├── image2.jpg 

	   │   │   └── ... 

	   │   ├── raio 

	   │   │   ├── image1.jpg 

	   │   │   ├── image2.jpg 

	   │   │   └── ... 

	   │   └── ... 

	   │ 

	   └── test 

	       ├── aro 

	       │   ├── image1.jpg 

	       │   ├── image2.jpg 

	       │   └── ... 

	       ├── cubo 

	       │   ├── image1.jpg 

	       │   ├── image2.jpg 

	       │   └── ... 

	       ├── raio 

	       │   ├── image1.jpg 

	       │   ├── image2.jpg 

	       │   └── ... 

	       └── ... 

	        

Deve ser observado também que alterar o dataset, pode se fazer necessário alterações nos parâmetros da arquitetura do modelo. 

	 

	As imagens serão redimensionadas para 254/254p, caso seja necessário a alteração desse parâmetro, deve se acessar o arquivo data_loader.py. 

	A alteração desse parâmetro implica diretamente na quantidade de features que deve ser setado na camada Linear do modelo, que se encontra em model.py 

	Deve-se também ser observado que a quantidade de classes deve ser igual a quantidade de outputs na camada linear do modelo (model.py). 

 
	

    Treinamento:
        Durante o treinamento, o modelo é alimentado com lotes de dados de treinamento, o numero de lotes pode ser configurado na string batche_size no arquivo config.json.
        Após cada lote, os gradientes são calculados e usados para atualizar os pesos do modelo com o otimizador.
        Isso ajuda o modelo a aprender os padrões nos dados de treinamento.
	
	Neste exemplo, foram usados diferentes tipos de configuracoes para gerar resultados.

    Validação:
        Após cada época, o modelo é testado em um conjunto de dados de validação separado.
        Os dados de validação não são usados no treinamento, portanto, são independentes e imparciais para avaliar o desempenho do modelo em novos dados.
        Durante a etapa de validação, o modelo gera previsões para os dados de validação e calcula a perda correspondente usando a função de perda definida.
        O valor médio dessa perda é chamado de validation loss (perda de validação) e é uma medida do quão bem o modelo está generalizando para dados não vistos.

    Acompanhamento do Treinamento:
        Ao longo das épocas de treinamento, você observará que a perda de treinamento diminui progressivamente à medida que o modelo se ajusta melhor aos dados de treinamento.
        A perda de validação, por outro lado, pode ter uma tendência semelhante no início, mas pode começar a aumentar se o modelo estiver sobreajustando aos dados de treinamento.
        Portanto, monitorar a perda de validação é uma maneira crucial de evitar o overfitting e garantir que o modelo esteja generalizando bem.
        
        Caso seja identificado overfitting atraves da avaliacao pos treinamento, pode se usar a opcao de data augmentation no arquivo config.json, que pode "enriquecer" o dataset com imagens em diferentes angulos e configuracoes de brilho/contraste.

    Escolha do Modelo Final:
        Ao final do treinamento, você pode comparar as curvas de perda de treinamento e de validação.
        Um bom modelo terá uma perda de validação baixa, indicando que ele generaliza bem para dados não vistos.

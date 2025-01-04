English Description Below

CNN-Mnist & Filter Project



Descrição

Este projeto envolve a implementação de duas partes: 

um exemplo simples de filtragem de imagens e um modelo de rede neural convolucional (CNN) para classificar o dataset MNIST. 

A seguir, apresento uma descrição de cada parte e o objetivo de cada implementação.

Filter

No primeiro arquivo, Filter.py, implementei uma operação de convolução simples usando a biblioteca TensorFlow. 

A ideia é aplicar dois filtros específicos em duas imagens de exemplo (uma imagem da China e outra de uma flor), visualizando os resultados da aplicação desses filtros em cada uma das imagens.

Carregamento das Imagens: 

As imagens são carregadas e normalizadas para o intervalo [0,1].

Definição dos Filtros: 

São criados dois filtros: 

um com uma linha central e o outro com uma coluna central.

Aplicação da Convolução: 

Utilizei a operação de convolução tf.nn.conv2d para aplicar os filtros às imagens, ajustando o stride e o padding para "SAME" (preservando o tamanho da imagem).

Visualização dos Resultados:

O resultado da convolução é exibido em uma figura com subplots, mostrando como cada filtro afeta a imagem.


CNN-Mnist


No segundo arquivo, CNN-Mnist.py, implementei um modelo de rede neural convolucional para classificar o dataset MNIST, que contém imagens de dígitos manuscritos. 

O objetivo é treinar uma CNN que seja capaz de classificar esses dígitos com alta precisão.

Pré-processamento dos Dados:

As imagens são carregadas e normalizadas para o intervalo [0, 1].

Os dados são divididos em conjuntos de treino, validação e teste.

Apliquei o escalonamento das imagens para uma melhor convergência do modelo.

Modelagem da Rede Neural:

O modelo é composto por várias camadas convolucionais, seguidas de camadas de normalização em lote (BatchNormalization), ativação ReLU, pooling e dropout para regularização.

A camada final utiliza softmax para realizar a classificação em 10 classes (dígitos de 0 a 9).

Treinamento:

O treinamento é feito com a otimização Nadam e a função de perda sparse_categorical_crossentropy.

Durante o treinamento, utilizei o agendador de taxa de aprendizado (LearningRateScheduler) para ajustar a taxa de aprendizado ao longo das épocas.

Além disso, foi incluído um callback para interromper o treinamento caso o modelo não melhore após um número determinado de épocas (EarlyStopping), além de salvar o melhor modelo (ModelCheckpoint).
Avaliação:

Ao final do treinamento, o modelo é avaliado no conjunto de teste, obtendo uma precisão de 99.54%.

 ━━━━━━━━━━━━━━━━━━━━ ━━━━━━━━━━━━━━━━━━━━ ━━━━━━━━━


CNN-Mnist & Filter Project

Description

This project involves the implementation of two parts:

A simple image filtering example and a Convolutional Neural Network (CNN) model to classify the MNIST dataset.

Below is a description of each part and the objective of each implementation.

Filter

In the first file, Filter.py, I implemented a simple convolution operation using the TensorFlow library.

The idea is to apply two specific filters to two sample images (one of China and the other of a flower) and visualize the results of applying these filters to each image.

Image Loading:

The images are loaded and normalized to the [0,1] range.

Filter Definition:

Two filters are created:

One with a central row.

The other with a central column.

Convolution Application:

I used the convolution operation tf.nn.conv2d to apply the filters to the images, adjusting the stride and padding to "SAME" (preserving the image size).

Result Visualization:

The result of the convolution is displayed in a figure with subplots, showing how each filter affects the image.

CNN-Mnist

In the second file, CNN-Mnist.py, I implemented a Convolutional Neural Network (CNN) model to classify the MNIST dataset, which contains images of handwritten digits.

The goal is to train a CNN capable of classifying these digits with high accuracy.

Data Preprocessing:

The images are loaded and normalized to the [0, 1] range.

The data is split into training, validation, and test sets.

Image scaling was applied for better model convergence.

Neural Network Modeling:

The model consists of several convolutional layers followed by batch normalization layers, ReLU activation, pooling, and dropout for regularization.

The final layer uses softmax to classify into 10 classes (digits from 0 to 9).

Training:

Training is done with the Nadam optimizer and the sparse categorical cross-entropy loss function.

During training, I used a learning rate scheduler (LearningRateScheduler) to adjust the learning rate over the epochs.

Additionally, a callback was included to stop training if the model doesn't improve after a certain number of epochs (EarlyStopping), as well as saving the best model (ModelCheckpoint).

Evaluation:

At the end of training, the model is evaluated on the test set, achieving an accuracy of 99.54%.

%GoogLeNet

clear all
clc

%% Preparazione dataset

%Metto dataset in un oggetto di tipo datastore

datasetPath = 'Dataset Taranto';
cropDS = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Split in datastore di train (le prime 70% immagini di ognuna delle due classi Pinna e Non Pinna)
% e validation (le rimanenti 30% immagini di ognuna delle due classi)
% randomized => scelta delle immagini in ordine casuale, altrimenti avviene
% in ordine alfabetico
[cropTrain,cropValidation] = splitEachLabel(cropDS,0.7,'randomized');


%% Inizializzazione CNN

%Carico la rete GoogLeNet
net = googlenet;

%Analizza la rete appena creata
analyzeNetwork(net)
%Ha 144 layer e accetta in input immagini 224x224 con 3 canali di colore

inputSize = net.Layers(1).InputSize;

%% Sostituzione last learnable layer e classification layer

% The convolutional layers of the network extract image features that the
% last learnable layer and the final classification layer use to classify
% the input image. These two layers, 'loss3-classifier' and 'output' in
% GoogLeNet, contain information on how to combine the features that the
% network extracts into class probabilities, a loss value, and predicted
% labels. To retrain a pretrained network to classify new images, replace
% these two layers with new layers adapted to the new data set.

% Layer graph from the trained network. If the network is a SeriesNetwork
% object, such as AlexNet, then convert the list of layers in net.Layers to
% a layer graph.
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

% Trova i layer da rimpiazzare (manualmente o, come fatto qui,
% automaticamente). 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

% Il learnableLayer spesso è un fully connected layer (cioè connette ogni
% suo neurone a ogni neurone del successivo layer, come nel multi-layer
% perceptron).
% Replace this fully connected layer with a new fully connected layer with
% the number of outputs equal to the number of classes in the new data set
% (5, in this example).
numClasses = numel(categories(cropTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
% In some networks, such as SqueezeNet, the last learnable layer is a
% 1-by-1 convolutional layer instead. In this case, replace the
% convolutional layer with a new convolutional layer with the number of
% filters equal to the number of classes. To learn faster in the new layer
% than in the transferred layers, increase the learning rate factors of the
% layer.
%This is not the case.
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

%Rimpiazza il learnable layer nel grafo
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% Crea un nuovo classification layer, che avrà 0 classi (le classi saranno
% automaticamente associate a questo layer in fase di trainNetwork)
newClassLayer = classificationLayer('Name','new_classoutput');

%Rimpiazza il learnable layer nel grafo
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% Plotta il nuovo grafo ottenuto
plot(lgraph); ylim([0,10]);


%% Freeze dei weigths dei primi 10 layer

layers = lgraph.Layers;
connections = lgraph.Connections;

% [Optional] Freezing the weights of many initial layers can significantly
% speed up network training. If the new data set is small, then freezing
% earlier network layers can also prevent those layers from overfitting to
% the new data set.

% Blocca il learning rate di pesi e bias nei primi 10 layer (the initial 'stem' of the GoogLeNet network)
layers(1:10) = freezeWeights(layers(1:10));
% Riconnetti tutti i layer nell'ordine originario
lgraph = createLgraphUsingConnections(layers,connections);


%% Re-addestramento GoogLeNet - Image pre-processing e augmentation

% ref: https://it.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html#TransferLearningUsingGoogLeNetExample-5

pixelRange = [-60 60];
angleRange = [-20,20];
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',angleRange, ...
    'RandXTranslation',pixelRange);

% PLOTTA UN ESEMPIO DI AUGMENTATION E METTI NELLA TESI

% creazione datastore aumentato passando oggetto imageDatastore + oggetto
% imageDataAugmenter
% N.B. Solo il training set è sottoposto ad Augmentation
% Il validation set è solo ridimensionato per essere compatibile alla
% dimensione di input della rete

cropAugmentedTrain = augmentedImageDatastore(inputSize(1:2),cropTrain,'DataAugmentation',augmenter);

cropAugmentedValidation = augmentedImageDatastore(inputSize(1:2),cropValidation);

%% TRAINING

miniBatchSize = 20; %vd SGD (stochastic gradient descent con mini-batch=20)
valFrequency = floor(numel(cropAugmentedTrain.Files)/miniBatchSize);    %num. di iterazioni in una epoch, quindi validazione alla fine di ogni epoch.
%NB: AGGIUNGI VARIABILE ENVIRONMENT PARALLEL O GPU
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', cropAugmentedValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'CheckpointPath','.\Checkpoint GoogLeNet');

TL_net = trainNetwork(cropAugmentedTrain,lgraph,options);


%% Classificazione immagini del validation set

[prediction,probs] = classify(TL_net,cropAugmentedValidation);
accuracy = mean(prediction == cropValidation.Labels)    %media di un vettore di 0 e 1


%% Matrice di confusione (anche detta tabella di errata classificazione)

plotConfusionMatrix(prediction,cropValidation.Labels)
%INSERIRE QUESTA IMMAGINE NELLA TESI
saveas(gcf,'confMat GoogLeNet.jpg');


%% Salvataggio

%Salvataggio workspace
save('workspace_googlenet.mat');   %classifica (in ordine) tutto il dataset di validation
%Salvataggio della rete addestrata
save('TL_googlenet.mat','TL_net');


%% Test sul validation dataset

for k = 1 : 3

indice = randperm(numel(cropValidation.Files),20);
figure('units','normalized','outerposition',[0 0 1 1]) %figure
for i = 1:20
    subplot(5,4,i)
    I = readimage(cropValidation,indice(i));    %legge immagine i-esima (in ordine) dal dataset di validation
    imshow(I)
    label = prediction(indice(i));
    title(string(label) + ", " + num2str(100*max(probs(indice(i),:)),3) + "%");
end
    %Salva i-esima immagine esemplificativa
    concat = strcat('Test GoogLeNet numero ',num2str(k),'.jpg');
    saveas(gcf,concat);
    
end
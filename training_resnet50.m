%ResNet-50

clear all
clc

%% Preparazione dataset

%Metto dataset in un oggetto di tipo datastore

datasetPath = 'Dataset Taranto';
cropDS = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Split in datastore di train
[cropTrain,cropValidation] = splitEachLabel(cropDS,0.7,'randomized');


%% Inizializzazione CNN

%Carico la rete ResNet-50
net = resnet50;

%Analizza la rete appena creata
analyzeNetwork(net)
% Ha 177 layers in totale, che corrispondono a una residual network di 50
% layer; accetta in input immagini 224x224 con 3 canali di colore

% Dimensioni immagine di input
inputSize = net.Layers(1).InputSize;

%numClasses: numero di categorie di classificazione (2)
numClasses = numel(categories(cropTrain.Labels));

lgraph = layerGraph(net);

% learnableLayer = lgraph.Layers(175);
% softmaxLayer = lgraph.Layers(175);
% classLayer = lgraph.Layers(177);

% Layer rimpiazzanti
newLearnableLayer = fullyConnectedLayer(numClasses,'Name','new_fc1000','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
newSoftmaxLayer = softmaxLayer('Name','new_fc1000_softmax');   %calcola la probabilità per ogni classe
newClassLayer = classificationLayer('Name','new_ClassificationLayer_fc1000');	%calcola la cross-entropy loss Li

% Rimpiazzo degli ultimi 3 layer
lgraph = replaceLayer(lgraph,lgraph.Layers(175).Name,newLearnableLayer);
lgraph = replaceLayer(lgraph,lgraph.Layers(176).Name,newSoftmaxLayer);
lgraph = replaceLayer(lgraph,lgraph.Layers(177).Name,newClassLayer);

% Plotta il nuovo grafo ottenuto
plot(lgraph); ylim([0,10]);


%% Freeze dei weigths del primo layer convoluzionale (il secondo - ESPERIMENTO)

layers = lgraph.Layers;
connections = lgraph.Connections;


% Blocca il learning rate di pesi e bias nei primi 10 layer (the initial 'stem' of the ResNet network)
layers(1:2) = freezeWeights(layers(1:2));
% Riconnetti tutti i layer nell'ordine originario
lgraph = createLgraphUsingConnections(layers,connections);  %PERMETTE DI AGGIRARE LA SOLA LETTURA DEGLI ATTRIBUTI


%% Re-addestramento ResNet-50 - Image pre-processing e augmentation

pixelRange = [-60 60];
angleRange = [-20,20];
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',angleRange, ...
    'RandXTranslation',pixelRange);

% PLOTTA UN ESEMPIO DI AUGMENTATION E METTI NELLA TESI

%Training set aumentato e ridimensionato 224x224
cropAugmentedTrain = augmentedImageDatastore(inputSize(1:2),cropTrain,'DataAugmentation',augmenter);
%Validation set ridimensionato 224x224
cropAugmentedValidation = augmentedImageDatastore(inputSize(1:2),cropValidation);


%% TRAINING

% When performing transfer learning, you do not need to train for as many epochs.
% ref: https://it.mathworks.com/help/deeplearning/examples/transfer-learning-using-alexnet.html

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
    'ExecutionEnvironment','parallel', ...
    'CheckpointPath','.\Checkpoint ResNet-50');

TL_net = trainNetwork(cropAugmentedTrain,lgraph,options);


%% Classificazione immagini del validation set

[prediction,probs] = classify(TL_net,cropAugmentedValidation);
accuracy = mean(prediction == cropValidation.Labels)    %media di un vettore di 0 e 1


%% Matrice di confusione (anche detta tabella di errata classificazione)

plotConfusionMatrix(prediction,cropValidation.Labels)
%INSERIRE QUESTA IMMAGINE NELLA TESI
saveas(gcf,'confMat ResNet50.jpg');

%% Salvataggio

%Salvataggio workspace
save('workspace_resnet50.mat');   %classifica (in ordine) tutto il dataset di validation
%Salvataggio della rete addestrata
save('TL_resnet50.mat','TL_net');


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
    concat = strcat('Test ResNet50 numero ',num2str(k),'.jpg');
    saveas(gcf,concat);
    
end
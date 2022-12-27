%AlexNet

clear all
clc

%% Preparazione dataset

% Metto dataset con i crop in un oggetto di tipo datastore

datasetPath = 'Dataset Taranto';
cropDS = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Split in datastore di train e di validazione
[cropTrain,cropValidation] = splitEachLabel(cropDS,0.7,'randomized');


%% Inizializzazione CNN

%Carico la rete AlexNet
net = alexnet;

%Analizza la rete appena creata
analyzeNetwork(net)
%Ha 25 layer e accetta in input immagini 227x227 con 3 canali di colore

inputSize = net.Layers(1).InputSize;

% ESTRAZIONE DEI LAYER DA TRASFERIRE
% Gli ultimi tre layer della AlexNet pre-addestrata sono configurati per
% 1000 classi. Questi tre layer devono essere ri-configurati per il nuovo
% problema di classificazione (=> per il nuovo dataset). Questi layer sono:
% 1) 'fc8' FC layer --> last learnable layer (=ultimo layer provvisto di
% parametri addestrabili, e quindi di learning rate)
% 2) 'prob' Softmax layer (sta per probabilities; applica la softmax function)
% 3) 'output' Classification Output Layer (calcola la cross-entropy loss)

% Estrae tutti i layer tranne gli ultimi tre (che saranno da sostituire),
% quindi i primi 22
layersTransfer = net.Layers(1:end-3);

%numClasses: numero di categorie del nuovo problema di classificazione (2)
numClasses = numel(categories(cropTrain.Labels));


%% Freeze dei weigths dei primi 2 layer convoluzionali (il secondo e il sesto - ESPERIMENTO)

layersTransfer(2).WeightLearnRateFactor = 0;
layersTransfer(2).BiasLearnRateFactor = 0;

layersTransfer(6).WeightLearnRateFactor = 0;
layersTransfer(6).BiasLearnRateFactor = 0;


%% Ricostruzione della rete

%layers è il nuovo "grafo" della rete, da addestrare
layers = [
    
    layersTransfer                              %i primi 22
    
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,'Name','new_fc8'),    %nota il lr abbastanza alto
    softmaxLayer('Name','new_prob'),            %calcola la probabilità per ogni classe
    classificationLayer('Name','new_output')];	%calcola la cross-entropy loss Li

%NB: i nuovi layer introdotti inferiscono il numero di classi (OutputSize) dall'OutputSize del layer precedente (il. For example, to specify the number of classes K of the
%network, include a fully connected layer with output size K and a softmax
%layer before the classification layer.

% NB:
% -il nuovo Fully Connected Layer ha OutputSize=numClasses=2 e
% InputSize='auto', cioè in fase di trainNetwork legge l'OutputSize del
% Fully Connected Layer a lui precedente (ci sono in questo caso i layer
% ReLU e Dropout tra i due Fully Connected in questione), che nel caso di
% AlexNet è 4096.
% -il nuovo Classification Layer ha OutputSize='auto', letto in fase di
% trainNetwork dall'OutputSize=numClasses=2 del Fully Connected Layer a lui
% precedente, e anche Classes='auto', impostato a ['No Pinna','Pinna'] dal
% cropAugmentedValidation (in qualche modo...)


%% Re-addestramento AlexNet - Image pre-processing e augmentation

pixelRange = [-60 60];
angleRange = [-20,20];
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',angleRange, ...
    'RandXTranslation',pixelRange);

% PLOTTA UN ESEMPIO DI AUGMENTATION E METTI NELLA TESI

%Training set aumentato e ridimensionato 227x227
cropAugmentedTrain = augmentedImageDatastore(inputSize(1:2),cropTrain,'DataAugmentation',augmenter);
%Validation set ridimensionato 227x227
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
    'CheckpointPath','.\Checkpoint AlexNet');

TL_net = trainNetwork(cropAugmentedTrain,layers,options);


%% Classificazione immagini del validation set

[prediction,probs] = classify(TL_net,cropAugmentedValidation);
accuracy = mean(prediction == cropValidation.Labels)    %media di un vettore di 0 e 1


%% Matrice di confusione (anche detta tabella di errata classificazione)

plotConfusionMatrix(prediction,cropValidation.Labels)
%INSERIRE QUESTA IMMAGINE NELLA TESI
saveas(gcf,'confMat AlexNet.jpg');


%% Salvataggio

%Salvataggio workspace
save('workspace_alexnet.mat');   %classifica (in ordine) tutto il dataset di validation
%Salvataggio della rete addestrata
save('TL_alexnet.mat','TL_net');


%% Test sul validation dataset
augmented = read(cropAugmentedValidation)
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
    concat = strcat('Test AlexNet numero ',num2str(k),'.jpg');
    saveas(gcf,concat);
    
end
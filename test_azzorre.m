clear
clc

%% DESCRIZIONE
% Questo programma effettua le seguenti operazioni:
% 1. test sul dataset delle azzorre con AlexNet, GoogLeNet, ResNet18
% 2. creazione di un file excel con due colonne, contenenti ciascuna gli
%    indirizzi delle immagini classificate come 'Pinna' e come 'No Pinna'


%% INIZIALIZZAZIONE DATASET

datasetPath = 'D:\Dati utente\Desktop\Tesi\Lavoro\Dataset Azzorre';
datasetDS = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% CARICAMENTO RETI + TEST + OUTPUT FILE EXCEL

netsList_temp = dir('TL_*');
% Escludi reti da non usare nel major voting; se non usi questa parte di
% codice allora chiama la variabile qui sopra netsList (cioè togli il _temp)
j=1;
for i=1:numel(netsList_temp)
    if string(netsList_temp(i).name)~="TL_googlenet_mb10.mat" & string(netsList_temp(i).name)~="TL_resnet50.mat"
        netsList(j) = netsList_temp(i);
        j=j+1;
    end
end
        

n = numel(netsList);

for i=1:n
    %Caricamento della rete i-esima
    net_toExtract = load(netsList(i).name);
    name_asCellArray = fieldnames(net_toExtract);
    name = name_asCellArray{1};
    net = net_toExtract.(name);
    
    %Ridimensionamento del dataset all'inputSize della rete i-esima
    inputSize = net.Layers(1).InputSize;
    datasetDS_resize = augmentedImageDatastore(inputSize(1:2),datasetDS);
    
    %Classificazione con rete i-esima
    [prediction,probs] = classify(net,datasetDS_resize);
    %predictionProbs = (max(probs')');
    accuracy = mean(prediction == datasetDS.Labels)

    %Salvataggio risultati della rete i-esima in un file excel
    results = table(datasetDS.Files,datasetDS.Labels,prediction,probs(:,1),probs(:,2), ...
        'VariableNames',{'Crop','TrueClass','Prediction','Prob_No_Pinna','Prob_Pinna'});
    netName = netsList(i).name(4:end-4);
    outputExcel = ['Risultati Azzorre ',netName,'.xls'];
    writetable(results,outputExcel);
    
    %Stampa matrice di confusione relativa alla rete i-esima
    plotConfusionMatrix(prediction,datasetDS.Labels)
    saveas(gcf,['confMat Azzorre ',netName,'.png'])
end


%% alcuni esempi di classificazione

for k = 1 : n

    indice = randperm(numel(datasetDS.Files),20);
    figure('units','normalized','outerposition',[0 0 1 1]) %figure
    for i = 1:20
        subplot(5,4,i)
        I = readimage(datasetDS,indice(i));    %legge immagine i-esima (in ordine) dal dataset di validation
        imshow(I)
        label = prediction(indice(i));
        title(string(label) + ", " + num2str(100*max(probs(indice(i),:)),3) + "%");
    end
    %Salva i-esima immagine esemplificativa
    netName = netsList(k).name(4:end-4);
    concat = ['Test Azzorre ',netName,' numero',num2str(k),'.jpg'];
    saveas(gcf,concat);
    
end

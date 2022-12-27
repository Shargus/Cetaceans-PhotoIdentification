clear
clc

%% DESCRIZIONE - MAJOR VOTING
% Reti "votanti": AlexNet, GoogLeNet, ResNet18
% Ogni rete ha eguale peso nella decisione finale
% Immagine classificata come Pinna se almeno due delle tre reti la
% classificano come Pinna.


%% LETTURA FILE EXCEL
T1 = readtable('Risultati Azzorre alexnet.xls');
T2 = readtable('Risultati Azzorre googlenet.xls');
T3 = readtable('Risultati Azzorre resnet18.xls');

trueClass = categorical(table2array(T1(:,2)));  % vere etichette
v1 = categorical(table2array(T1(:,3)));         % etichette date da alexnet
v2 = categorical(table2array(T2(:,3)));         % etichette date da googlenet
v3 = categorical(table2array(T3(:,3)));         % etichette date da resnet18
p1_np = double(table2array(T1(:,4)));           % probabilità no pinna alexnet
p1_p = double(table2array(T1(:,5)));            % probabilità pinna alexnet
p2_np = double(table2array(T2(:,4)));           % ...
p2_p = double(table2array(T2(:,5)));
p3_np = double(table2array(T3(:,4)));
p3_p = double(table2array(T3(:,5)));

%Probabilità medie (tra le 3 reti) delle classi Pinna e No Pinna
%(model averaging)
probsPinna = mean([p1_p,p2_p,p3_p]')';

% tol: tolleranza, cioè minima probabilità accettata per 'Pinna', nell'HMV
% 0.0 < tol < 1.0
tol = 0.97;

%votes sono i voti per Pinna (servono per l'hard major voting)
votes = sum(([v1,v2,v3]=='Pinna')')';


%% VOTAZIONE
for i=1:size(votes,1)
    
    % soft major voting classico
    if probsPinna(i)>0.5
        predictionSoft{i} = 'Pinna';
        probsSoft(i) = probsPinna(i);
    else
        predictionSoft{i} = 'No Pinna';
        probsSoft(i) = 1-probsPinna(i);
    end
    
    % hard major voting con minima tolleranza per la probabilità media di Pinna
    if votes(i)>=2 & probsPinna(i)>tol %OPPURE p1_p(i)>tol & p2_p(i)>tol & p3_p(i)>tol
        predictionHard{i} = 'Pinna';
        probsHard(i) = probsPinna(i);
    else
        predictionHard{i} = 'No Pinna';
        probsHard(i) = 1-probsPinna(i);
    end
    
end
predictionSoft = categorical(predictionSoft');
predictionHard = categorical(predictionHard');
probsSoft = probsSoft';
probsHard = probsHard';


%% SALVATAGGIO RISULTATI IN FILE EXCEL

resultsSoft = table(predictionSoft,probsSoft,'VariableNames',{'Prediction','Accuracy'});
writetable(resultsSoft,'Risultati Azzorre soft major voting.xls');

resultsHard = table(predictionHard,probsHard,'VariableNames',{'Prediction','Accuracy'});
writetable(resultsHard,'Risultati Azzorre hard major voting.xls');


%% MATRICE DI CONFUSIONE

plotConfusionMatrix(predictionSoft,trueClass)
saveas(gcf,'confMat Azzorre soft major voting.png')

plotConfusionMatrix(predictionHard,trueClass)
saveas(gcf,'confMat Azzorre hard major voting.png')


%% alcuni esempi di classificazione

datasetPath = 'C:\Users\tommy\Desktop\Materiale Tesi\Lavoro\Dataset Azzorre';
datasetDS = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

for k = 1 : 6

indice = randperm(numel(datasetDS.Files),20);
figure('units','normalized','outerposition',[0 0 1 1]) %figure

% HMVT
for i = 1:15
    subplot(5,3,i)
    I = readimage(datasetDS,indice(i));    %legge immagine i-esima (in ordine) dal dataset di validation
    imshow(I)
    label = predictionSoft(indice(i));
    title(string(label),'FontSize',20); % + ", " + num2str(100*max(probsHard(indice(i),:)),3) + "%");
end
    %Salva i-esima immagine esemplificativa
    concat = strcat('Test Azzorre HMVT numero ',num2str(k),'.png');
    saveas(gcf,concat);
    
end
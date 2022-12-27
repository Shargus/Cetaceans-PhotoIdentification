function plotConfusionMatrix(predictedLabels,trueLabels)
%Compute and plot the confusion matrix for validation dataset

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(trueLabels,predictedLabels,'FontSize',15);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

end

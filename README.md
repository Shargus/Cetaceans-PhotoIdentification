# Cetaceans photo-identification

Binary image classifier to check whether an image depicts a dolphin's fin or not.

**Data augmentation** is applied (random horizontal reflections, horizontal translations, rotations).

Four **CNNs** are trained (AlexNet, GoogLeNet, ResNet-18, ResNet-50) through **transfer learning**, on a dataset of photos taken in the Gulf of Taranto.

The final prediction is based on a majority voting scheme (**ensemble learning**).

## User guide
- **training_alexnet.m**, **training_googlenet.m**, **training_resnet18.m**, **training_resnet50.m**: code for training the corresponding classification models
- **majority_voting.m**: code for performing hard/soft majority voting
- **test_azzorre.m**: code for testing the trained CNNs on another dataset (consisting of photos taken near the Azores)
- **createLgraphUsingConnections.m**: function for creating a layer graph, with specified layers and connections between them
- **findLayersToReplace.m**: function for finding the last learnable layer and classification layer
- **freezeWeights.m**: function for setting the learning rates of the specified layersto zero
- **plotConfusionMatrix.m**: function for plotting the confusion matrix, given the arrays of predicted and ground-truth labels

clear all
warning off


%###########network and data initialization###############

pathP = 'percentile24IMG';%here you have to save the 3-channels images, not the original images

imP = imageDatastore(pathP, ...
                     'IncludeSubfolders', true, ...
                     'LabelSource','foldername');
               
%[imdsTrain,imdsTest] = splitEachLabel(imP,0.8,'randomized'); %split test and training set
imP = shuffle(imP)

%sarebbe da andare a dividere (secondo la teoria, magari c'è una funzione
%che lo fa da sola) il train in 4 parti uguali. A turno, ciascuna di queste
%verrà utilizzata come validation mentre le altre 3 come training. Bisogna
%capire come dire alla rete di fare questo, ovvero come capire di cambiare.

net = alexnet;  %load AlexNet
siz=[227 227];

%######Dividing into k folds#########

k=4;
partStores{k} = [];
for i = 1:k
    temp = partition(imP,k,i);
    partStores{i} = temp.Files;
end

idx = crossvalind('Kfold',k,k);
netTransfer{k} = [];


final_pred = categorical([]);
%###########tuning rete############
for i = 1:4

    test_idx = (idx == i);
    train_idx = ~test_idx;

    test_store = imageDatastore(partStores{test_idx},'IncludeSubfolders',true,'LabelSource','foldernames');
    train_store = imageDatastore(cat(1, partStores{train_idx}),'IncludeSubfolders',true,'LabelSource','foldernames');
    
    numTr = size(train_store.Files); %number of patterns in the training set

    numClasses = numel(categories(train_store.Labels)); %number of classes in the training set
    
    %############resizing images############

    YTest = test_store.Labels; %save test set labels
    test_store = augmentedImageDatastore(siz,test_store);    
    train_store = augmentedImageDatastore(siz,train_store);
    
    layersTransfer = net.Layers(1:end-3);
    layers = [
            layersTransfer
            fullyConnectedLayer(numClasses,'WeightLearnRateFactor',50,'BiasLearnRateFactor',50)
            softmaxLayer
            classificationLayer];

    miniBatchSize = 30;
    learningRate = 1e-4;
    metodoOptim='sgdm';
    options = trainingOptions(metodoOptim,...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',30,...
        'InitialLearnRate',learningRate,...
        'ExecutionEnvironment','parallel',...
        'Verbose',false,...
        'Plots','training-progress');
    numIterationsPerEpoch = floor(numTr/miniBatchSize);
    
    %############training############

    netTransfer = trainNetwork(train_store,layers,options);
    
    [YPred,scores] = classify(netTransfer,test_store);
    valid_accuracy(i) = mean(YPred == YTest);
    confusionchart(YTest,YPred);
    
    pred = cat(2, test_store.Files, YPred, YTest);   % NOME FILE   PREDICTION  LABEL
    
    %SCORES FOLD1
    %SCORES FOLD2
    %SCORES FOLD3
    final_pred = [final_pred;pred];
    
end

filename = [pathP '.xlsx'];  %Quando alleni per la seconda volta fai [pathP '2' '.xlsx']
writematrix(final_pred, filename);

avg_valid_accuracy = mean(valid_accuracy);
avg_valid_accuracy

%{
%############test#############

YTest = imdsTest.Labels;
[YPred,scores] = classify(netTransfer,imdsTest);


%############data############
accuracy = mean(YPred == YTest);
accuracy
confusionchart(YTest,YPred)
%}
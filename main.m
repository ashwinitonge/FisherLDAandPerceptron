%-----------------------------------------
%   Author: Parisa, Ashwini
%   Function: Main
%   Pupose: Evaluate Fisher LDA and Perceptron
%-----------------------------------------

warning('off');

%Input Parameters
NumberOfSamples = 1000; %Number Of samples to generate for each class.
NumberOfFeatures = 2;   %Number Of Features 
NumberOfClasses = 2;    %Number Of Classes
% first class vs all classes will be used to train and test the models.

%Mean for each class.
Mean = zeros(NumberOfClasses); 
Mean(1) = 3; 
Mean(2) = 10;
Std = 0.5; %Standard deviation for all classes.
NumberOfFold = 3; %Number of folds.

%Generate Instances.
Instances = DataGen(NumberOfClasses, NumberOfFeatures, NumberOfSamples, Mean, Std);

TotalInstances = zeros(2*NumberOfSamples, NumberOfFeatures);
TotalLabels = zeros(2*NumberOfSamples, 1);

%First class considered as +
inst = 1;
for i=1:NumberOfSamples
  TotalInstances(inst,:) =  Instances(i,:,1);
  TotalLabels(inst) = 1;
  inst = inst + 1;
end

%Rest of the classes as -
for c=2:NumberOfClasses
    for i=1:NumberOfSamples
      TotalInstances(inst,:) =  Instances(i,:,c);
      TotalLabels(inst) = -1;
      inst = inst + 1;
    end
end

%%k-fold cross validation
Part = cvpartition(TotalInstances(:,end),'k',NumberOfFold);

TotalAccf = 0.0;
TotalSentivityf = 0.0;
TotalSpecifityf = 0.0;

TotalAccp = 0.0;
TotalSentivityp = 0.0;
TotalSpecifityp = 0.0;

%Iterate through each fold
for p=1:NumberOfFold  

FoldTrainInstances = zeros(Part.TrainSize(p) , NumberOfFeatures);
FoldTrainLabels = zeros(Part.TrainSize(p), 1);
FoldTestInstances = zeros(Part.TestSize(p), NumberOfFeatures);
FoldTestLabels = zeros(Part.TestSize(p), 1);

%Genrate train and test samples for each fold
Train = training(Part, p);
Test = test(Part, p);
TrainItr = 1;
TestItr = 1;
for i=1:2*NumberOfSamples
    if Train(i) ~= 0
    FoldTrainInstances(TrainItr,:) =  TotalInstances(i,:) ;
    FoldTrainLabels(TrainItr) = TotalLabels(i);
    TrainItr = TrainItr + 1;
    end
    if Test(i) ~= 0
    FoldTestInstances(TestItr,:) =  TotalInstances(i,:) ;
    FoldTestLabels(TestItr) = TotalLabels(i);
    TestItr = TestItr + 1;
    end   
end

%FisherLDA
[v, c1min, c1max, c2min, c2max] = trainFisherLDA (FoldTrainInstances, FoldTrainLabels);
[TPf,FNf,FPf,TNf] = testFisherLDA(FoldTestInstances, FoldTestLabels, v, c1min, c1max, c2min, c2max );
TotalAccf = TotalAccf + ((TPf + TNf)/(TPf + FNf + FPf + TNf));
TotalSentivityf = TotalSentivityf + ((TPf)/(TPf + FNf));
TotalSpecifityf = TotalSpecifityf + ((TNf)/(TNf + FPf));

%Perceptron
[Model_weights,Model_bias] = trainPerceptron(FoldTrainInstances, FoldTrainLabels, NumberOfFeatures);
[TPp,FNp,FPp,TNp] = testPerceptron(FoldTestInstances, FoldTestLabels, Model_weights,Model_bias );
TotalAccp = TotalAccp + ((TPp + TNp)/(TPp + FNp + FPp + TNp));
TotalSentivityp = TotalSentivityp + ((TPp)/(TPp + FNp));
TotalSpecifityp = TotalSpecifityp + ((TNp)/(TNp + FPp));

%print per fold result
fprintf('--------------Results For Fold - %d ------------\n', p); 
fprintf('Projection vector : \n');
fprintf( '%f\t', v);
fprintf('\nModel Weights: \n');
fprintf('%f\t', Model_weights);
fprintf('\nModel bias: %f\n', Model_bias);
fprintf('---------Confusion Matrix Fisher LDA-------------\n');
fprintf('\t\t\t\t\t\tPredicted\n');
fprintf('\t\t\t\t + class\t\t - class \n');
fprintf('Actual + class\t\t%d\t\t\t%d\n',TPf, FNf);
fprintf('Actual - class\t\t%d\t\t\t%d\n',FPf, TNf);
fprintf('---------Confusion Matrix Perceptron-------------\n');
fprintf('\t\t\t\t\t\tPredicted\n');
fprintf('\t\t\t\t + class\t\t - class \n');
fprintf('Actual + class\t\t%d\t\t\t%d\n',TPp, FNp);
fprintf('Actual - class\t\t%d\t\t\t%d\n',FPp, TNp);
end

%Calculate Accuracy, Sensitivity and Specificity
Accf = (TotalAccp / NumberOfFold) * 100;
Sentivityf = TotalSentivityf / NumberOfFold;
Specifityf = TotalSpecifityf / NumberOfFold;

Accp = (TotalAccp / NumberOfFold) * 100;
Sentivityp = TotalSentivityp / NumberOfFold;
Specifityp = TotalSpecifityp / NumberOfFold;

%Print Accuracy, Sensitivity and Specificity
fprintf('----------------------Overall Results-------------------\n');
fprintf('Classifier\tAccuracy\tSensitivity\tSpecificity\n');
fprintf('FisherLDA\t%f\t%f\t%f\n',Accf,Sentivityf,Specifityf);
fprintf('Perceptron\t%f\t%f\t%f\n',Accp,Sentivityp,Specifityp);


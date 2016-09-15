%-----------------------------------------
%   Author: Parisa, Ashwini
%   Function: DataGen
%   Pupose: Generate instances.
%-----------------------------------------

function Instances = DataGen(NumberOfClasses, NumberOfFeatures, NumberOfSamples, Mean, Std)
Instances = zeros(NumberOfSamples, NumberOfFeatures,NumberOfClasses);

%generate instances randomly for particular distribution for each class.
for m=1:NumberOfClasses
sample = normrnd(Mean(m),Std,[NumberOfSamples,NumberOfFeatures]);
Instances(:,:,m) = sample;
end
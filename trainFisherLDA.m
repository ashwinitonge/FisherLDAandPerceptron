%-----------------------------------------
%   Author: Parisa, Ashwini
%   Function: trainFisherLDA
%   Pupose: Train Fisher LDA model.
%-----------------------------------------

function [v, c1min, c1max, c2min, c2max ] = trainFisherLDA (Instances, Labels)

c1temp = Labels>0;
c1 = Instances(c1temp,:);
c2temp = Labels<0;
c2 = Instances(c2temp,:);

scatter(c1(:,1),c1(:,2),6,'r'),hold on;
scatter(c2(:,1),c2(:,2),6,'b');

% Number of observations of each class
n1=size(c1,1);
n2=size(c2,1);

%Mean of each class
mu1=mean(c1);
mu2=mean(c2);

% Average of the mean of all classes
mu=(mu1+mu2)/2;

% Center the data (data-mean)
d1=c1-repmat(mu1,size(c1,1),1);
d2=c2-repmat(mu2,size(c2,1),1);

% Calculate the within class variance (SW)
s1=d1'*d1;
s2=d2'*d2;
sw=s1+s2;
invsw=inv(sw);

%Projection Vector
v=invsw*transpose(mu1-mu2);

%Project Data
y2=c2*v;
y1=c1*v;

c1min = min(y1);
c1max = max(y1);

c2min = min(y2);
c2max = max(y2);
end



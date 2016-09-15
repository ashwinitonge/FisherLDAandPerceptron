%-----------------------------------------
%   Author: Parisa, Ashwini
%   Function: testFisherLDA
%   Pupose: Test Fisher LDA model.
%-----------------------------------------

function [c1c1,c1c2,c2c1,c2c2] = testFisherLDA(Instances, Labels, v, c1min, c1max, c2min, c2max )
c1c1 = 0;
c1c2 = 0;
c2c1 = 0;
c2c2 = 0;
NSample = size(Instances,1);

%Estimate decision point from projected data.
min = c2min;
if c1min > c2min
    min = c1min;
end    

max = c2max;
if c1max < c2max
    max = c1max;
end 

mean = (min + max)/2;

  

for i=1:NSample
    %Get projection for test sample.
    y = Instances(i,:)*v;
    
    %Predict class label
    if (y > mean)
       if c1min > mean
           Out = 1;
       else
           Out = -1;               
       end
    else
       if c1min < mean
           Out = 1;
       else
           Out = -1;   
       end
    end
    
    %Classification result
    if ((Out == Labels(i)))
        if (Out == 1)
            c1c1 = c1c1 + 1;
        else
            c2c2 = c2c2 + 1;
        end

    else        
        if (Out == 1)
            c2c1 = c2c1 + 1;
        else
            c1c2 = c1c2 + 1;
        end 
    end
end
%-----------------------------------------
%   Author: Parisa, Ashwini
%   Function: testPerceptron
%   Pupose: Test Perceptron model.
%-----------------------------------------

function [c1c1,c1c2,c2c1,c2c2] = testPerceptron(Instances, Labels, Model_weights,Model_bias )
c1c1 = 0;
c1c2 = 0;
c2c1 = 0;
c2c2 = 0;
NSample = size(Instances,1);
ACROSS = 2;

%Iterate through each sample
for i=1:NSample
 
    inputObservation = Instances(i, : );
    
    %Perceptron output for sample
    perceptronOutput = sum( Model_weights .* inputObservation, ACROSS ) + Model_bias;
    
    %Predict label
    out = 1;
    if (perceptronOutput < 0)
        out = -1;
    end
    
    %Classification result
    if ((out == Labels(i)))
        if (out == 1)
            c1c1 = c1c1 + 1;
        else
            c2c2 = c2c2 + 1;
        end

    else        
        if (out == 1)
            c2c1 = c2c1 + 1;
        else
            c1c2 = c1c2 + 1;
        end 
    end

end


end
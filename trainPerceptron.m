%-----------------------------------------
%   Author: Parisa, Ashwini
%   Function: trainPerceptron
%   Pupose: Train Perceptron model.
%-----------------------------------------

function [Model_weights,Model_bias] = trainPerceptron(Instances, Labels, NumberOfFeatures)

    ACROSS = 2;

    Model_weights = zeros(1, NumberOfFeatures);
    Model_bias    = 0.0;

    maxNumSteps = 1000;
    NumberOfSamples = size(Instances,1);
    
    maxNorm    = realmin;
    for iObservation = 1:NumberOfSamples
        observationNorm = norm( Instances(iObservation,:) );
        if observationNorm > maxNorm
            maxNorm = observationNorm;
        end
    end
    enclosingBallRadius        = maxNorm;
    enclosingBallRadiusSquared = enclosingBallRadius .^ 2;
    
    %Iterate through samples
    for iStep = 1:maxNumSteps

        isAnyObsMisclassified = false;

        for iObservation = 1:NumberOfSamples;
            
            inputObservation = Instances( iObservation, : );
            desiredLabel     = Labels(iObservation); % +1 or -1

            perceptronOutput = sum( Model_weights .* inputObservation, ACROSS ) + Model_bias;
            margin           = desiredLabel * perceptronOutput;

            isCorrectLabel   = margin > 0;

            % If he model misclassifies the observation, update the
            % weights and the bias.

            if ~isCorrectLabel

                isAnyObsMisclassified = true;

                weightCorrection = desiredLabel  * inputObservation;
                Model_weights    = Model_weights + weightCorrection;
                
                biasCorrection   = desiredLabel .* enclosingBallRadiusSquared;
                Model_bias       = Model_bias   + biasCorrection;                
            end 
            
        end 

        if ~isAnyObsMisclassified
            disp( 'Done!' );
            break;
        end

    end 

end

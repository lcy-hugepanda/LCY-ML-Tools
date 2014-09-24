% given a trained PRTools mapping and a test dataset
% return the confusion matrix of this dataset that classified by the given trained mapping
% imput argument: (1) mapping: a trained mapping
% imput argument: (2) A: the test dataset

% output argument: cm : the confusion matrix of the classification result,
% element in row i and column j of comfusion matrix cm is the number of samples with true
% label i but classified as class j

function cm =LC_ConfusionMatrix (mapping, A)
    if(~isa(mapping, 'prmapping') )
        fprintf('parameter error!\n');
        cm=-1;
        return;
    else
        [sampleCount,featureCount,classCount]=getsize(A);
        [feat,class]=size(mapping);
        
        if (featureCount ~= feat) || (classCount ~= class)
            fprintf('parameter error!\n');
            cm = -1;
            return ;
        end
        
        labels=getnlab(A);        
        cm = zeros(classCount,classCount);

        [~,predictedLabels]= max(getdata(A*mapping),[],2);        
        for i=1:1:classCount
            labMasks=ones(sampleCount,1).*i;
            
            for j=1:1:classCount
                resMasks=ones(sampleCount,1).*j;
                cm (i,j) = sum ( ((labels==labMasks) & (predictedLabels==resMasks)));
            end
        end
        
    end    
end
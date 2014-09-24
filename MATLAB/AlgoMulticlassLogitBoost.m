% LogitBoost for multiclass classification Version 1.0
% LCY-seso Novermber 27th 2013


function out =  LC_MulticlassLogitBoost (varargin)

    %% set default parameters
    argin = setdefaults(varargin,[],1);

    %% Execution Path A£º construct an untrained mapping
    if mapping_task(argin,'definition')
        out = define_mapping(argin,'untrained',['MulticlassLogitBoost' int2str(argin{2})]);

    %% Execution Path A£¬triaining a multiclass LogitBoost classifier
    elseif mapping_task(argin,'training')

        % get imput arguments for LogitBoost
        [A, param] = deal(argin{:});

        iterationCount=param;
        [sampleCount, ~, classCount]=getsize(A);
        labels=getnlab(A); 

        y=zeros(sampleCount,classCount);
        for i=1:1:sampleCount
            y(i,labels(i,1))=1;
        end

        % initialization
        baseClassifiers = cell (classCount,iterationCount);            

        px =ones (sampleCount,classCount)./classCount;
        weights=ones (sampleCount,classCount).*((classCount-1)/(classCount*classCount));
        workingResponse=(y-px)./(px.*(ones(sampleCount,classCount)-px));

        predictTmp=zeros(sampleCount,classCount,iterationCount);

        for i=1:1:iterationCount
            fprintf('Iteration %d: \n',i);

            % fit the Jth regression tree for each class sepeartely
            for j=1:1:classCount            
                % use regression tree from Matlab toolkit
                baseClassifiers{j,i}=...
                    classregtree(A.data,workingResponse(:,j),'weights',weights(:,j));

                predictTmp (:,j,i) =eval (baseClassifiers{j,i}, A.data);
            end

            % update working response and  sample weights for next iteration
            px= exp(  predictTmp(:,:,i)  )./...
                ( repmat ( sum ( exp( predictTmp(:,:,i) ),2),1,classCount));
            weights=px.*(1-px);
            workingResponse=(y-px)./(px.*(ones(sampleCount,classCount)-px));

            workingResponse( find (workingResponse>4))=4;
            workingResponse( find (workingResponse<-4))=-4;
            weights(find (workingResponse< 1e-04)) =2*1e-03;

        end

        % construct trained mapping
        data.baseClassifiers = baseClassifiers;
        data.iterationCount = iterationCount;
        out = trained_classifier(A, data);

    %% Execution Path C: testing
    elseif mapping_task(argin,'execution')
        [A,w] = deal ( argin{1:2}) ; 
        v = getdata(w);
        
        [sampleCount,~]=size(A);
        [~,classCount]=size(w);
        
        fitValues=zeros(sampleCount,classCount);
        
        for i=1:1: v.iterationCount
            for j=1:1: classCount
                fitValues(:,j) = fitValues(:,j)+ eval(v.baseClassifiers{j,i},A.data);
            end
        end
        
        prob= exp (fitValues )./...
                ( repmat ( sum ( exp( fitValues ),2),1,classCount));
        
        out = setdat(A,prob,w);

    %% Handle error
    else
      error('Illegal call')  
    end
return

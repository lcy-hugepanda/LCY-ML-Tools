% AW_LibsvmOCSVM ���ڰ�װLibSVM�⣨��OCSVM���ಿ�֣�
% ʵ���Ͻ�����һ����װ������ĿǰΪֹ��

% ���ߣ����ҳ�
% ����ʱ�䣺2013��4��24��10:48:13

function out = OCLT_AlgoLibsvmOCSVM(varargin)
    name = 'LibSVM OC-SVM Classifier';
	%prtrace(mfilename);

    argin = setdefaults(varargin,[],' ',0.1,[]);
	
	if mapping_task(argin,'definition')
        %% Path A: ������δѵ���ģ�untrained��������
        %   ��������㷨��ʱ��û���ṩ��������û���ṩѵ�����ݼ������ߴ˴����߼���
		w = prmapping(mfilename,'untrained');
		w = setname(w,name);
		out = w;
        %fprintf('Untrained Subspace Classifier\n');
    else
        %% Path B: ѵ��������
        %   �����ṩ��ѵ�����ݼ���ͬʱparam����һ���Ѿ�ѵ���õķ������ĵ���
        %   ����Ϊ��ѵ��������������A*AW_OCAdaBoost, A*AW_OCAdaBoost(A)
        if mapping_task(argin,'training') % �ؼ���֧A������һ����װ
            [A, W, rejf, validA] = deal(argin{:});
            
            [~,numAttributes,numClasses] = getsize(A); 
            [label, inst] =  DataConvertLibsvm2PRTools(A);
            data.libsvmArgs = W;
            if (length(W) > 0) % No Auto Model Selection
                data.libsvmModel = svmtrain(label, inst, ...
                    ['-s 2 -t 2', ' -n ', num2str(rejf), ' ', data.libsvmArgs]);                 
            else
                [~,bestg] = OCLT_LibsvmModelSelectionForOCSVM(A, rejf, validA);
                %msgbox(['Best G is ', num2str(bestg)]);
                data.libsvmModel = svmtrain(label, inst, ...
                    ['-s 2 -t 2', ' -n ', num2str(rejf), ' -g ', num2str(bestg), ' ', data.libsvmArgs]);               
            end

            data.SV = A(data.libsvmModel.sv_indices, :);
           
            out = prmapping(mfilename, 'trained' , data, getlablist(A),numAttributes,numClasses);
            out = setname(out,name);
        elseif mapping_task(argin,'execution') % �ؼ���֧B�����Է�����
            %% Path C: ����������
            % ʹ�ò������ݼ������������ĵ��÷�ʽʵ���������֣� 
            %       evaResult = AW_OCAdaBoost(testingData, W)
            %       evaResult = testingData * W;  % �Զ��ж�mapping����
            %   �����ʱ�򶼰��յ�һ�֣�����
            %       ����1���������ݼ�
            %       ����2��ѵ���õ�mapping
            %       ����ֵ�����dataset
            %   ��ν�ġ����dataset����һ�������dataset������data��֮�⣬������
            %   ��������ݣ�����1��һ�¡�data���Ǻ�����ʾ���������s x c������
            %       s��������
            %       c�������
            %   �ɼ��������dataset����data���У�ÿһ�б�ʾһ�������ڸ�����ϵĹ���
            %   ���յķ��������ݲ�����soft������������һ����
            %   ���ԣ�����ͨ�����·�ʽ�ӡ����dataset����ȡ�÷�����������
            %    [mx,result] = max(+evaResult,[],2); % result�ǽ��
               [A,W] = deal(argin{1:2});
            %fprintf('Evaluating Subspace Classifier...\n');
           
            if(isnumeric(A)) % Sometimes, A is a double array rather than a dataset
                A = dataset(A);
                [~, inst] = DataConvertLibsvm2PRTools(A);
            else
                [~, inst] = DataConvertLibsvm2PRTools(A);
            end
            
            [numInstances,numAttributes,numClasses] = getsize(A); 
            [label_true, inst_true] = DataConvertLibsvm2PRTools(A);
            
            [predict_label, ~, dec_values] = ...
                svmpredict(label_true, inst, W.data.libsvmModel);
            
            data = zeros(numInstances, 2);
            data(:, 2) = dec_values;
                      
            out = prdataset(A); 
            out.data = data;
            out.featlab = A.lablist{1,1};
            %out = setdata(out, data);
        else
            out = 'error';
            error('Wrong Invoking, please check input arguments.');
        end
	end

	return
end


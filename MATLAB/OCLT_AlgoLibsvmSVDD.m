% AW_LibsvmSVDD ���ڰ�װLibSVM�⣨��SVDD���ಿ�֣�
% ʵ���Ͻ�����һ����װ������ĿǰΪֹ��

% ����ʾ����
% LC_LibsvmSVDD(tr, ' ', 0.05, tr); tr����������ѡ�����������tr��ʹ��Ĭ�ϲ������ɵڶ�����������

% ���ߣ����ҳ�
% ����ʱ�䣺2013��5��9��8:38:41

function out = OCLT_AlgoLibsvmSVDD(varargin)
    name = 'LibSVM SVDD Classifier';
	%prtrace(mfilename);

    % Set default parameters
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
            % LibSVM��SVDD�ƺ��Ὣ���е����������޲����Ϊ����������������Ҫ�������
            % target_class(A)������A������ֱ������Ļ��ᵼ��mapping������ռ�ά����1��Ӱ���������
            % ����޸�һ�£��������Ȼ��A��ֻ����ѵ����ʱ��ȡһ��target_class         
            [A, W, rejf, validA] = deal(argin{:});
            
            [~,numAttributes,numClasses] = getsize(A); 
            if(1 ~= numClasses)
                A = target_class(A);
            end
            
            [label, inst] = DataConvertLibsvm2PRTools(A);
            data.libsvmArgs = W;
            if (nargin < 4) % No Auto model selection
                data.libsvmModel = svmtrain(label, inst,['-s 2', ' -n ', num2str(rejf), ' ', data.libsvmArgs]);                  
            else 
                [~,bestg] = OCLT_LibsvmModelSelectionForSVDD(A, rejf, validA, 0);
                %msgbox(['Best G is ', num2str(bestg)]);
                data.libsvmModel = svmtrain(label, inst, ...
                    ['-s 5 -t 2', ' -n ', num2str(rejf), ' -g ', num2str(bestg), ' ', data.libsvmArgs]);                
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
                A = prdataset(A);
                [label_true, inst] = DataConvertLibsvm2PRTools(A);
            else
                [label_true, inst] = DataConvertLibsvm2PRTools(A);
            end
            
            [numInstances,numAttributes,numClasses] = getsize(A); 
            
            [predict_label, ~, dec_values] = ...
                svmpredict(label_true, inst, W.data.libsvmModel);
            
            data = zeros(numInstances, 2);
            % for -s 5 (SVDD), dec_values is not the same as regular
            % SVM. It contains the distance between the sample point and the
            % center of the circle, so dec_values should be taken down by
            % the radius of the optimized hypersphere
            %data(:, 2) =  W.data.libsvmModel.radius - dec_values;            
            data(:, 1) =  dec_values - W.data.libsvmModel.radius;
            data(find(abs(data(:,1)) < 1e-3), 1) = -1;
            
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


% AW_LibsvmC ���ڰ�װLibSVM�⣨�ķ��ಿ�֣�
% ʵ���Ͻ�����һ����װ������ĿǰΪֹ��

% ������ʹ��rsscc����(PRTools)�������ӿռ�OCC
% ���ߣ����ҳ�
% ����ʱ�䣺2013��4��24��10:48:13

function out = LC_LibsvmC(varargin)
    name = 'LibSVM Classifier';
	%prtrace(mfilename);

    argin = setdefaults(varargin,[],' ');
	
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
            [A, W] = deal(argin{:});
            
            [numInstances,numAttributes,numClasses] = getsize(A); 
            [label, inst] = DataConvertLibsvm2PRTools(A);
            data.libsvmArgs = W;
            data.libsvmModel = svmtrain(label, inst, data.libsvmArgs);
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
            % for -s 5 (SVDD), dec_values is not the same as regular
            % SVM. It contains the distance between the sampe point and the
            % center of the circle, so dec_values should be taken down by
            % the radius of the optimized hypersphere
            if(strcmp(W.data.libsvmArgs(4),'5')) % for SVDD
                data(:, 2) = dec_values - W.data.libsvmModel.radius;
            else
                data(:, 2) = dec_values;
            end
            
            
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


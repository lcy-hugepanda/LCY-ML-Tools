% libsvmread�������ARFF�ļ���ʽ�����ݶ�ȡ�ӿ�
% ������dataset������Ϊ�˳������ԭ�еĴ���dataset�ķ���
function [label, inst] = LCY_DataConvertLibsvm2PRTools(A)

[instanceCount,featureCount,classCount] = getsize(A); 

% label���֣�instanceCount x 1 ������
label  = getnlab(A);
    % �����OCC�����ݼ������TargetŪ��+1, OutlierŪ��-1
    if(length(A.lablist{1,1})>0 && ~strcmp(A.lablist{1,1}(1,:), 'outlier')) % lablist is {'outlier', 'target'}
        label(find(label==1)) = 1;
        label(find(label==2)) = -1;
    elseif (length(A.lablist{1,1})>0 && ~strcmp(A.lablist{1,1}(1,:), 'target'))
        label(find(label==1)) = -1;
        label(find(label==2)) = 1;        
    end


% inst���֣�instanceCount x featureCount ��sparse����
% inst(i,j)��ʾ��i�������ĵ�j��featureֵ
% P.S. �����ݼ��ϴ��ʱ��ǰ���3��zeros���������ڴ治���״�����������̫��Ļ����ļ���תһ��
data = +A;
%if(instanceCount*featureCount < 3000*500 )
    i = zeros(instanceCount * featureCount, 1);
    j = zeros(instanceCount * featureCount, 1);
    s = zeros(instanceCount * featureCount, 1);
    for t = 1 : 1 : featureCount
        i(((t-1)*instanceCount + 1) : t*instanceCount) = 1:1:instanceCount;
        j(((t-1)*instanceCount + 1) : t*instanceCount) = t;
        s(((t-1)*instanceCount + 1) : t*instanceCount) = data(:,t);
    end
    inst = sparse(i, j, s, instanceCount, featureCount);
%else % ʹ���ļ���ת
%    LC_DataDatasetSaveToLibSVM(A, 'D:\LC_tempFile');
%end


end
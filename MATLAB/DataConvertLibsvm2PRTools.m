% libsvmread函数针对ARFF文件格式的数据读取接口
% 参数是dataset，这是为了充分利用原有的处理dataset的方法
function [label, inst] = LCY_DataConvertLibsvm2PRTools(A)

[instanceCount,featureCount,classCount] = getsize(A); 

% label部分，instanceCount x 1 的向量
label  = getnlab(A);
    % 如果是OCC的数据集，则把Target弄成+1, Outlier弄成-1
    if(length(A.lablist{1,1})>0 && ~strcmp(A.lablist{1,1}(1,:), 'outlier')) % lablist is {'outlier', 'target'}
        label(find(label==1)) = 1;
        label(find(label==2)) = -1;
    elseif (length(A.lablist{1,1})>0 && ~strcmp(A.lablist{1,1}(1,:), 'target'))
        label(find(label==1)) = -1;
        label(find(label==2)) = 1;        
    end


% inst部分，instanceCount x featureCount 的sparse矩阵
% inst(i,j)表示第i个样本的第j个feature值
% P.S. 当数据集较大的时候，前面的3个zeros矩阵会出现内存不足的状况，所以如果太大的话用文件中转一下
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
%else % 使用文件中转
%    LC_DataDatasetSaveToLibSVM(A, 'D:\LC_tempFile');
%end


end
% 用于命令行显示指定数据集的特性

function DataInfo(A)

label = getnlab(A);
label_idx = unique(label);
fprintf('数据集包含%d个类别\n',length(label_idx));
label_list = A.lablist{1,1};
for i = 1 : 1 : length(label_idx)
    fprintf('类别%s包含%d个样本\n',label_list(i,:),length(find(label==i)));
end


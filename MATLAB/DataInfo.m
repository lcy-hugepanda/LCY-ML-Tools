% ������������ʾָ�����ݼ�������

function DataInfo(A)

label = getnlab(A);
label_idx = unique(label);
fprintf('���ݼ�����%d�����\n',length(label_idx));
label_list = A.lablist{1,1};
for i = 1 : 1 : length(label_idx)
    fprintf('���%s����%d������\n',label_list(i,:),length(find(label==i)));
end


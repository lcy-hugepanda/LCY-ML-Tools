% ������dataset��PRTools��ʽ���ϲ�
% ������ͬ���ľ������Ը�����������

function A = OCLT_DataCombineDatasets(part1,part2)
    data1 = +part1;
    label1 = getnlab(part1);
    data2 = +part2;
    label2 = getnlab(part2) * 2;

    A = prdataset([data1 ; data2], [label1 ; label2]);
    A = oc_set(A,'2');
end
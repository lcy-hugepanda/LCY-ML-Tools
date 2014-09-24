% [���ĺ���]��·�����벢�������

% �����߼����� 2012��9��21��10:32:13 ���ҳ�
%   ���ں��������Ҫ����ʮ��������֤�����ݼ��������Գ���ѵ�����Ͳ��Լ���
%   ��֣�����Ҫ�ṩδ�������ԭʼ���ݼ���

% BUG FIXED 2013��1��10��8:38:59 ���ҳ�
%   �ڰ��ձ�����ֹ����У����ܴ�����Ϊ����������ɵ�����������һ������

% UPDATED 2013��11��6��9:26:02 ���ҳ�
%	������prtools 5֮�������������߼�Ҳ��Ҫ�޸�

function [trainingPRTool, testingPRTool, origPRTool, trainingWeka,testingWeka, origWeka]=dataForPRToolsAndWeka(arffFilePath,splitingRate)
%[trainingPRTool, testingPRTool, trainingWeka,testingWeka]=dataForPRToolsAndWeka(arffFilePath,splitingRate)
%��arffFilePathָ����·���ж�ȡ���ݣ�ͬʱ����splitingRateָ���Ļ���ѵ��������������
%��ԭʼ���ݼ���splitingRate��Ϊѵ�����ݣ�1-splitingRate����������Ϊ��������
%trainingPRTool��testingPRTool�ֱ��ǿɱ�PRTool���߰�ʹ�õ�ѵ���������ϲ���������
%trainingWeka��testingWeak�ֲ��ǿɱ�Wekaʹ�õ�ѵ�������������������
%origPRTool��origWeka��δ����ֵ�ԭʼ���ݼ�

data=loadARFF(arffFilePath);
data.sort(data.classAttribute());
origWeka=data;

totalCount=data.numInstances();
attributeCount=data.numAttributes();
classCount=data.numClasses();

%��weka�����������������һλ�������
%for i=1: 1: totalCount
     totalLabels=data.attributeToDoubleArray(attributeCount-1);
%end

% ԭʼ���ݼ�����������
nbOrigDataset=zeros(classCount,1);


%nbTrainingSet��¼��ѵ����������ÿһ����Ҫ���ٸ�����
nbTrainingSet=zeros(classCount,1);
nbTrainTotal=round(totalCount*splitingRate);

%����ԭѵ����������ÿһ����������Ŀ
for i=1: 1: classCount
    tmp=ones(totalCount,1)*(i-1);
    nbOrigDataset(i,1)=sum(~(tmp~=totalLabels));
    if(i==classCount)
        nbTrainingSet(i,1)=nbTrainTotal-sum(nbTrainingSet);
    else
        nbTrainingSet(i,1)=round(nbOrigDataset(i,1)*splitingRate);
    end
end


if(1 ~= splitingRate)
    % nbTestingSet��¼�Ų�����������ÿһ����Ҫ���ٸ�����
    % nbTestingSet=nbOrigDataset-nbTrainingSet;
    %origWeka=javaObject('weka.core.Instances', data, int32(totalCount));
    trainingWeka=javaObject('weka.core.Instances', data, int32(nbTrainTotal));
    testingWeka= javaObject('weka.core.Instances', data, int32(totalCount-nbTrainTotal));

    base=0;
    for i=1: 1: classCount;
        labs=randperm(nbOrigDataset(i,1))'-ones(nbOrigDataset(i,1),1)+base;
        base=base+nbOrigDataset(i,1);

        trainCount=1;
        %��ȡ��i��ѵ��������wekaʹ�õ�ѵ��������
        while(trainCount<=nbTrainingSet(i,1))
            trainingWeka.add(data.instance(labs(trainCount,1)));
            trainCount=trainCount+1;
        end

        %��ȡ��i�����������wekaʹ�õĲ���������
        while(trainCount<=nbOrigDataset(i,1))
            testingWeka.add(data.instance(labs(trainCount,1)));
            trainCount=trainCount+1;
        end

    end

    %����PRTToolsʹ�õ�ѵ��������
    vectors=zeros(nbTrainTotal,attributeCount-1);
    features=cell(attributeCount-1,1);



    %��trainingWeak�г�ȡ����
    for i=1:1:attributeCount-1
        % ����nbTrainTotal��vector�������������ܲ�ͬ�����ʱ����������ɣ�
        % ���������ر���һ�£��������
        thisAttribute = testingWeka.attributeToDoubleArray(i-1);
        [vectorLength,~] = size(vectors);
        if (length(thisAttribute) > vectorLength)
            vectors(:,i)= thisAttribute(1: vectorLength );
        elseif (length(thisAttribute) < vectorLength)
            vectors=zeros(length(thisAttribute),attributeCount-1);
            [vectorLength,~] = size(vectors);
            vectors(:,i)= thisAttribute;
        else
            vectors(:,i)= thisAttribute;
        end
        %�����������
        features(i,1)=cellstr((trainingWeka.attribute(i-1).name().toCharArray())');
    end
    feaLabels=char(features);

    classes=cell(classCount,1);

    attr=trainingWeka.classAttribute();
    %��ȡ�������
    for i=1: 1: classCount
        classes(i,1)=cellstr((attr.value(i-1).toCharArray()'));
    end

    %��weka�����������������һλ�������
    labels=trainingWeka.attributeToDoubleArray(attributeCount-1);
    labels = labels(1:vectorLength);

    prior=zeros(classCount,1);
    for n=1: 1: classCount
        prior(n,1)=nbTrainingSet(n,1)/nbTrainTotal;
    end

    %classes = char(classes);

    trainingPRTool=prdataset(vectors,labels);
    trainingPRTool=setfeatlab(trainingPRTool,feaLabels);
    trainingPRTool=setlablist(trainingPRTool,classes);
    trainingPRTool=setprior(trainingPRTool,prior);

    %����PRTToolsʹ�õĲ���������
    vectors=zeros(totalCount-nbTrainTotal,attributeCount-1);
    features=cell(attributeCount-1,1);

    %��testingWeka�г�ȡ����
    for i=1:1:attributeCount-1
        thisAttribute = testingWeka.attributeToDoubleArray(i-1);
        [vectorLength,~] = size(vectors);
        if (length(thisAttribute) > vectorLength)
            vectors(:,i)= thisAttribute(1: vectorLength );
        elseif (length(thisAttribute) < vectorLength)
            vectors=zeros(length(thisAttribute),attributeCount-1);
            [vectorLength,~] = size(vectors);
            vectors(:,i)= thisAttribute;
        else
            vectors(:,i)= thisAttribute;
        end

        %�����������
        features(i,1)=cellstr((testingWeka.attribute(i-1).name().toCharArray())');
    end
    feaLabels=char(features);

    %��ȡ�������
    classes=cell(classCount,1);
    attr=testingWeka.classAttribute();
    for i=1: 1: classCount
        classes(i,1)=cellstr((attr.value(i-1).toCharArray())');
    end

    %��weka�����������������һλ�������
    labels=testingWeka.attributeToDoubleArray(attributeCount-1);
    labels = labels(1:vectorLength);

    prior=zeros(classCount,1);
    for n=1: 1: classCount
        prior(n,1)=(nbOrigDataset(n,1)-nbTrainingSet(n,1))/(totalCount-nbTrainTotal);
    end

    testingPRTool=prdataset(vectors,labels);
    testingPRTool=setfeatlab(testingPRTool,feaLabels);
    testingPRTool=setlablist(testingPRTool,classes);
    testingPRTool=setprior(testingPRTool,prior);
else
    % DUMMY routine, for the spliting rate 1
end


%����PRTToolsʹ�õĲ���������
vectors=zeros(totalCount,attributeCount-1);
features=cell(attributeCount-1,1);

%��origWeka�еõ�ԭʼ���ݼ���PRTools���ݼ�
for i=1:1:attributeCount-1
    vectors(:,i)=origWeka.attributeToDoubleArray(i-1);
    %�����������
    features(i,1)=cellstr((origWeka.attribute(i-1).name().toCharArray())');
end
feaLabels=char(features);

%��ȡ�������
classes=cell(classCount,1);
attr=origWeka.classAttribute();
for i=1: 1: classCount
    classes(i,1)=cellstr((attr.value(i-1).toCharArray())');
end

%��weka�����������������һλ�������
labels=origWeka.attributeToDoubleArray(attributeCount-1);


prior=zeros(classCount,1);
for n=1: 1: classCount
    prior(n,1)=(nbOrigDataset(n,1))/(totalCount);
end

origPRTool=prdataset(vectors,labels);
origPRTool=setfeatlab(origPRTool,feaLabels);
origPRTool=setlablist(origPRTool,classes);
origPRTool=setprior(origPRTool,prior);

if( 1 == splitingRate)
	% For the case of NON-SPLIT, all return datasets are the whole dataset
    % (Add by Jiachen Liu)
    trainingPRTool = origPRTool;
    testingPRTool = origPRTool;
    trainingWeka = origWeka;
    testingWeka = origWeka;
end
return
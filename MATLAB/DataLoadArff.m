% [姐姐的函数]从路径读入并拆分数据

% 程序逻辑更新 2012年9月21日10:32:13 刘家辰
%   由于后面可能需要进行十迭交叉验证等数据集处理，所以除了训练集和测试集的
%   拆分，还需要提供未经处理的原始数据集。

% BUG FIXED 2013年1月10日8:38:59 刘家辰
%   在按照比例拆分过程中，可能存在因为四舍五入造成的样本个数不一致问题

% UPDATED 2013年11月6日9:26:02 刘家辰
%	升级到prtools 5之后，这里面的相关逻辑也需要修改

function [trainingPRTool, testingPRTool, origPRTool, trainingWeka,testingWeka, origWeka]=dataForPRToolsAndWeka(arffFilePath,splitingRate)
%[trainingPRTool, testingPRTool, trainingWeka,testingWeka]=dataForPRToolsAndWeka(arffFilePath,splitingRate)
%从arffFilePath指定的路径中读取数据，同时按照splitingRate指定的划分训练样本与检测样本
%将原始数据集的splitingRate化为训练数据，1-splitingRate的样本划分为测试数据
%trainingPRTool、testingPRTool分别是可被PRTool工具包使用的训练样本集合测试样本集
%trainingWeka、testingWeak分布是可被Weka使用的训练样本集与测试样本集
%origPRTool、origWeka是未经拆分的原始数据集

data=loadARFF(arffFilePath);
data.sort(data.classAttribute());
origWeka=data;

totalCount=data.numInstances();
attributeCount=data.numAttributes();
classCount=data.numClasses();

%在weka中最后决策向量的最后一位是类别标号
%for i=1: 1: totalCount
     totalLabels=data.attributeToDoubleArray(attributeCount-1);
%end

% 原始数据集的样本个数
nbOrigDataset=zeros(classCount,1);


%nbTrainingSet记录着训练样本集中每一类需要多少个样本
nbTrainingSet=zeros(classCount,1);
nbTrainTotal=round(totalCount*splitingRate);

%计算原训练样本集中每一类样本的数目
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
    % nbTestingSet记录着测试样本集中每一类需要多少个样本
    % nbTestingSet=nbOrigDataset-nbTrainingSet;
    %origWeka=javaObject('weka.core.Instances', data, int32(totalCount));
    trainingWeka=javaObject('weka.core.Instances', data, int32(nbTrainTotal));
    testingWeka= javaObject('weka.core.Instances', data, int32(totalCount-nbTrainTotal));

    base=0;
    for i=1: 1: classCount;
        labs=randperm(nbOrigDataset(i,1))'-ones(nbOrigDataset(i,1),1)+base;
        base=base+nbOrigDataset(i,1);

        trainCount=1;
        %抽取第i类训练样本到weka使用的训练样本集
        while(trainCount<=nbTrainingSet(i,1))
            trainingWeka.add(data.instance(labs(trainCount,1)));
            trainCount=trainCount+1;
        end

        %抽取第i类测试样本到weka使用的测试样本集
        while(trainCount<=nbOrigDataset(i,1))
            testingWeka.add(data.instance(labs(trainCount,1)));
            trainCount=trainCount+1;
        end

    end

    %生成PRTTools使用的训练样本集
    vectors=zeros(nbTrainTotal,attributeCount-1);
    features=cell(attributeCount-1,1);



    %从trainingWeak中抽取数据
    for i=1:1:attributeCount-1
        % 由于nbTrainTotal与vector的样本个数可能不同（拆分时四舍五入造成）
        % 所以这里特别处理一下，避免出错。
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
        %获得属性名称
        features(i,1)=cellstr((trainingWeka.attribute(i-1).name().toCharArray())');
    end
    feaLabels=char(features);

    classes=cell(classCount,1);

    attr=trainingWeka.classAttribute();
    %获取类别名称
    for i=1: 1: classCount
        classes(i,1)=cellstr((attr.value(i-1).toCharArray()'));
    end

    %在weka中最后决策向量的最后一位是类别标号
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

    %生成PRTTools使用的测试样本集
    vectors=zeros(totalCount-nbTrainTotal,attributeCount-1);
    features=cell(attributeCount-1,1);

    %从testingWeka中抽取数据
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

        %获得属性名称
        features(i,1)=cellstr((testingWeka.attribute(i-1).name().toCharArray())');
    end
    feaLabels=char(features);

    %获取类别名称
    classes=cell(classCount,1);
    attr=testingWeka.classAttribute();
    for i=1: 1: classCount
        classes(i,1)=cellstr((attr.value(i-1).toCharArray())');
    end

    %在weka中最后决策向量的最后一位是类别标号
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


%生成PRTTools使用的测试样本集
vectors=zeros(totalCount,attributeCount-1);
features=cell(attributeCount-1,1);

%从origWeka中得到原始数据集的PRTools数据集
for i=1:1:attributeCount-1
    vectors(:,i)=origWeka.attributeToDoubleArray(i-1);
    %获得属性名称
    features(i,1)=cellstr((origWeka.attribute(i-1).name().toCharArray())');
end
feaLabels=char(features);

%获取类别名称
classes=cell(classCount,1);
attr=origWeka.classAttribute();
for i=1: 1: classCount
    classes(i,1)=cellstr((attr.value(i-1).toCharArray())');
end

%在weka中最后决策向量的最后一位是类别标号
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
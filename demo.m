% 交叉验证代码

clc;
clear;

%变量定义%
allFaces=[];
peopleNum=90;
onesFacesNum=26;
trainNum=20;
width=40;
height=50;

tags=[];
knn_max_k=5;
%图片读取%
for i=1:peopleNum
    for j=1:onesFacesNum
        if(i<10)
            allFaces=[allFaces,reshape(imread(strcat('database\AR_Gray_50by40\AR00',num2str(i),'-',num2str(j),'.tif')),[width*height,1])];
        else
            allFaces=[allFaces,reshape(imread(strcat('database\AR_Gray_50by40\AR0',num2str(i),'-',num2str(j),'.tif')),[width*height,1])];
        end
    end
end
allFaces =double(allFaces);%转型浮点，避免运算损失train_datatrain_data
%打TAG%
for i=1:peopleNum
    for j=1:onesFacesNum
        tags=[tags,i];
    end
end

%重复及随机试验%
numExperiments = 10; % 重复实验次数
numSamples = [ 6, 10, 14, 18]; % 不同数量的训练样本

knnAccuracy = zeros(length(numSamples), numExperiments);
pcaAccuracy = zeros(length(numSamples), numExperiments);

for i = 1:numExperiments
    fprintf("第%i次实验：\n",i);
    for j = 1:length(numSamples)
        % 随机选择训练样本
        A=1:onesFacesNum;
        B=double(randperm(onesFacesNum,numSamples(j)));
        C = setdiff(A, B);
        train_data=[];
        train_labels=[];
        test_data=[];
        test_labels=[];
        for k = 1:peopleNum
            train_data=[train_data,allFaces(:,((k-1)*onesFacesNum)+B)];
            train_labels=[train_labels,tags(:,((k-1)*onesFacesNum)+B)];
            test_data=[test_data,allFaces(:,((k-1)*onesFacesNum)+C)];
            test_labels=[test_labels,tags(:,((k-1)*onesFacesNum)+C)];
        end


        eigenvectors=LPP(train_data,100055);


        % PCA 提取特征后进行 KNN 分类
        top_eig_vec = eigenvectors(:, 1:50);  % 选择前pca_n个特征向量作为主成分
        % 数据投影
        pcatrain_data= top_eig_vec'*train_data;
        pcatest_data= top_eig_vec'*test_data    ;
        TruePreNum=0;
        for eachTestFace = 1:size(test_data,2)
            dist=[];
            for eachTrainFace = 1:size(train_data,2)
                dist=[dist,norm(pcatrain_data(:,eachTrainFace)-pcatest_data(:,eachTestFace))];
            end
            [~,I]=sort(dist);
            predictTag = mode(train_labels(I(1:knn_max_k)));%K近邻
            if(predictTag==test_labels(eachTestFace))
                TruePreNum=TruePreNum+1;
            end
        end
        pcaAccuracy(j, i) = TruePreNum/size(test_data,2);
    end
end

% 计算平均识别率和标准差

pcalppMeanAccuracy = mean(pcaAccuracy, 2);
pcalppStdDev = std(pcaAccuracy, 0, 2);
% 显示结果
comparisonTable = table( pcalppMeanAccuracy, pcalppStdDev, 'RowNames', {num2str(numSamples(1)), num2str(numSamples(2)), num2str(numSamples(3)), num2str(numSamples(4))});
disp(comparisonTable);

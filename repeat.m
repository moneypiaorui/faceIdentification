clc;
clear;

%变量定义%
allFaces=[];
train_data=[];
test_data=[];
peopleNum=90;
onesFacesNum=26;
trainNum=20;
width=40;
height=50;
% peopleNum=15;
% onesFacesNum=11;
% trainNum=6;
% width=80;
% height=100;

tags=[];
train_labels = [];
test_labels = [];

knn_max_k=5;
%pca_n=100;


%图片读取%

for i=1:peopleNum    
    for j=1:onesFacesNum     
        if(i<10)
           allFaces=[allFaces,reshape(imread(strcat('database\AR_Gray_50by40\AR00',num2str(i),'-',num2str(j),'.tif')),[width*height,1])];     
        else
            allFaces=[allFaces,reshape(imread(strcat('database\AR_Gray_50by40\AR0',num2str(i),'-',num2str(j),'.tif')),[width*height,1])];   
        end          

       % if(i<10)
       %    allFaces=[allFaces,reshape(imread(strcat('database\yale_face10080\subject0',num2str(i),'_',num2str(j),'.bmp')),[width*height,1])];     
       % else
       %     allFaces=[allFaces,reshape(imread(strcat('database\yale_face10080\subject',num2str(i),'_',num2str(j),'.bmp')),[width*height,1])];   
       % end   
        
    end
end
allFaces =double(allFaces);%转型浮点，避免运算损失

%打TAG%
for i=1:peopleNum    
    for j=1:onesFacesNum     
        tags=[tags,i];
    end
end

%取训练测试集%
trainIndex=(1:trainNum);
testIndex=(trainNum+1:onesFacesNum);

for i = 1:peopleNum
       train_data=[train_data,allFaces(:,((i-1)*onesFacesNum)+trainIndex)];
        train_labels=[train_labels,tags(:,((i-1)*onesFacesNum)+trainIndex)];
       test_data=[test_data,allFaces(:,((i-1)*onesFacesNum)+testIndex)];
       test_labels=[test_labels,tags(:,((i-1)*onesFacesNum)+testIndex)];
end

%PCA%
mean_data = mean(train_data,2);% 计算样本均值
centered_data = train_data - mean_data;% 中心化数据
cov_matrix = centered_data*centered_data';% 计算协方差矩阵
[eigenvectors, eigenvalues] = eig(cov_matrix);% 计算协方差矩阵的特征值和特征向量
[~, idx] = sort(diag(eigenvalues), 'descend');
eigenvectors=eigenvectors(:,idx);%特征向量按特征值排序

%knn
for pca_n=100:10:160
    for knn_k=1:knn_max_k
        top_eig_vec = eigenvectors(:, 1:pca_n);  % 选择前pca_n个特征向量作为主成分
        % 数据投影
       pcatrain_data= top_eig_vec'*(train_data-mean_data);
       pcatest_data= top_eig_vec'*(test_data-mean_data);
        TruePreNum=0;
        for eachTestFace = 1:size(test_data,2)
            dist=[];
            for eachTrainFace = 1:size(train_data,2)
                dist=[dist,norm(pcatrain_data(:,eachTrainFace)-pcatest_data(:,eachTestFace))];
            end
            [~,I]=sort(dist);
            predictTag = mode(train_labels(I(1:knn_k)));%K近邻
            if(predictTag==test_labels(eachTestFace))
                 TruePreNum=TruePreNum+1;
            end
        end
        fprintf("pca_n=%d KNN_K=%d 正确数=%d 正确率=%1f\n",pca_n,knn_k,TruePreNum,TruePreNum/size(test_data,2));
    end
end


%重复及随机试验%
numExperiments = 10; % 重复实验次数
numSamples = [ 3, 4, 5, 6]; % 不同数量的训练样本

knnAccuracy = zeros(length(numSamples), numExperiments);
pcaAccuracy = zeros(length(numSamples), numExperiments);

for i = 1:numExperiments
    fprintf("第%i次实验：\n",i);
    for j = 1:length(numSamples)
        % 随机选择训练样本
        A=1:onesFacesNum;
        % B=double(randperm(onesFacesNum,numSamples(j)));
        B=1:numSamples(j);
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

        %PCA%
        meanFace=mean(train_data,2);
        STDtrain_data=train_data-meanFace;%去中心化
        covMatrix = STDtrain_data*STDtrain_data';%协方差矩阵
        [eigenvectors, eigVal] = eig(covMatrix);% 计算特征值和特征向量
        [~, idx] = sort(diag(eigVal), 'descend');

        % KNN 分类
        pcatrain_data= (train_data-meanFace);
        pcatest_data=(test_data -meanFace);
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
        knnAccuracy(j, i) = TruePreNum/size(test_data,2);

        % PCA 提取特征后进行 KNN 分类
        top_eig_vec = eigenvectors(:, 1:160);  % 选择前pca_n个特征向量作为主成分
        % 数据投影
       pcatrain_data= top_eig_vec'*(train_data-meanFace);
       pcatest_data= top_eig_vec'*(test_data-meanFace);
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
knnMeanAccuracy = mean(knnAccuracy, 2);
knnStdDev = std(knnAccuracy, 0, 2);
pcaMeanAccuracy = mean(pcaAccuracy, 2);
pcaStdDev = std(pcaAccuracy, 0, 2);

% 显示结果
comparisonTable = table(knnMeanAccuracy, knnStdDev, pcaMeanAccuracy, pcaStdDev, 'RowNames', {'3', '4', '5', '6'});
disp(comparisonTable);

% 展示前10个特征脸
figure;
top_eig_vec = eigenvectors(:, 1:160); 
for i = 1:10
    subplot(2, 5, i);
    imshow(reshape(top_eig_vec(:, i),[height, width]), []);
    title(['eigFace ', num2str(i)]);
end
sgtitle('Top 10 Eigenfaces');

% 展示用前10个、前20、前50个投影进行重构原图生成的图片与误差
reconstructedFaces1 = meanFace + top_eig_vec(:, 1:10) * (top_eig_vec(:, 1:10)' *  (train_data-meanFace));
reconstructedFaces2 = meanFace + top_eig_vec(:, 1:20) * (top_eig_vec(:, 1:20)' *  (train_data-meanFace));
reconstructedFaces3 = meanFace + top_eig_vec(:, 1:50) * (top_eig_vec(:, 1:50)' *  (train_data-meanFace));
figure;
for i = 1:5
    subplot(3, 5, i);
    imshow(reshape(reconstructedFaces1(:, i), [height, width]), []);
    title('维度10');
    subplot(3, 5, 5+i);
    imshow(reshape(reconstructedFaces2(:, i), [height, width]), []);
    title('维度：20');
    subplot(3, 5, 10+i);
    imshow(reshape(reconstructedFaces3(:, i), [height, width]), []);
    title('维度50');
end
sgtitle('Reconstructed Faces using Top 10/20/50 Eigenfaces');

% 计算重构误差
reconstructionError = norm( (train_data-meanFace) - (top_eig_vec(:, 1:50) * (top_eig_vec(:, 1:50)' *  (train_data-meanFace))), 'fro') / norm( (train_data-meanFace), 'fro');
disp(['Reconstruction Error using Top 50 Eigenfaces: ', num2str(reconstructionError)]);


% 可视化展示
tem=allFaces-meanFace;
showGroups=5;
colors=rand(showGroups,3);


% 数据投影到2维空间
pca2D = eigenvectors(:, 1:2)' * tem(:,1:onesFacesNum*showGroups ); % 用前2个特征向量进行投影
figure;
for i=1:showGroups
    scatter(pca2D(1, (i-1)*onesFacesNum+1:i*onesFacesNum),pca2D(2, (i-1)*onesFacesNum+1:i*onesFacesNum),50, repmat(colors(i,:),  onesFacesNum,1), 'filled');
    hold on;
end
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('PCA Projection to 2D Space');

% 数据投影到3维空间
pca3D = eigenvectors(:,1:3)' * tem(:,1:onesFacesNum*showGroups ); % 用前3个特征向量进行投影
figure;
for i=1:showGroups
    scatter3(pca3D(1, (i-1)*onesFacesNum+1:i*onesFacesNum),pca3D(2, (i-1)*onesFacesNum+1:i*onesFacesNum),pca3D(3, (i-1)*onesFacesNum+1:i*onesFacesNum),50,repmat(colors(i,:),  onesFacesNum,1), 'filled');
    hold on;
end
xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');
title('PCA Projection to 3D Space');
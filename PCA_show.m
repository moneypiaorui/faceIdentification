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

pca_dim=50;%pca降维维度
%pca降维%

mean_data = mean(train_data,2);% 计算样本均值
centered_data = train_data - mean_data;% 中心化数据
cov_matrix = centered_data*centered_data';% 计算协方差矩阵
[eigenvectors, eigenvalues] = eig(cov_matrix);% 计算协方差矩阵的特征值和特征向量
[~, idx] = sort(diag(eigenvalues), 'descend');
eigenvectors=eigenvectors(:,idx);%特征向量按特征值排序

% 可视化展示%
tem=allFaces-mean_data;
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
pca3D = eigenvectors(:, 1:3)' * tem(:,1:onesFacesNum*showGroups ); % 用前3个特征向量进行投影
figure;
for i=1:showGroups
    scatter3(pca3D(1, (i-1)*onesFacesNum+1:i*onesFacesNum),pca3D(2, (i-1)*onesFacesNum+1:i*onesFacesNum),pca3D(3, (i-1)*onesFacesNum+1:i*onesFacesNum),50,repmat(colors(i,:),  onesFacesNum,1), 'filled');
    hold on;
end
xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');
title('PCA Projection to 3D Space');
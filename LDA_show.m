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


% LDA降维 %
% 计算每个类别的均值向量
num_classes = peopleNum; % 类别数
mean_vectors = zeros( size(train_data, 1),num_classes);
for i = 1:num_classes
    mean_vectors(:,i) = mean(train_data(:,train_labels == (i)),2);
end
% 计算类内散度矩阵
within_scatter_matrix = zeros(size(train_data, 1));
for i = 1:num_classes
    class_data = train_data(:,train_labels == (i));
    class_mean = mean_vectors(:,i);
    class_scatter =   (class_data - class_mean)*(class_data - class_mean)';
    within_scatter_matrix = within_scatter_matrix + class_scatter;
end
% 计算类间散度矩阵
overall_mean = mean(train_data,2);
between_scatter_matrix = zeros(size(train_data, 1));
for i = 1:num_classes
    class_mean = mean_vectors(:,i);
    between_scatter =  (class_mean - overall_mean)*(class_mean - overall_mean)';
    between_scatter_matrix = between_scatter_matrix + between_scatter;
end
% 计算广义特征值问题的解
[eigenvectors, eigenvalues] = eig(pinv(within_scatter_matrix) * between_scatter_matrix);
%特征向量按特征值排序
[~, idx] = sort(diag(eigenvalues), 'descend');
eigenvectors=eigenvectors(:,idx);
% 选择前k个特征向量作为主成分
LDA_dim = 50; % 设置降维后的维度
selected_eigenvectors = eigenvectors(:, end:-1:end-LDA_dim+1);
projected_data =  selected_eigenvectors'*train_data;% 降维投影

% 可视化展示%
tem=allFaces;
showGroups=5;
colors=rand(showGroups,3);

% 数据投影到2维空间
lda2D = eigenvectors(:, 1:2)' * tem(:,1:onesFacesNum*showGroups ); % 用前2个特征向量进行投影
figure;
for i=1:showGroups
    scatter(lda2D(1, (i-1)*onesFacesNum+1:i*onesFacesNum),lda2D(2, (i-1)*onesFacesNum+1:i*onesFacesNum),50, repmat(colors(i,:),  onesFacesNum,1), 'filled');
    hold on;
end
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('LDA Projection to 2D Space');

% 数据投影到3维空间
lda3D = eigenvectors(:, 1:3)' * tem(:,1:onesFacesNum*showGroups ); % 用前3个特征向量进行投影
figure;
for i=1:showGroups
    scatter3(lda3D(1, (i-1)*onesFacesNum+1:i*onesFacesNum),lda3D(2, (i-1)*onesFacesNum+1:i*onesFacesNum),lda3D(3, (i-1)*onesFacesNum+1:i*onesFacesNum),50,repmat(colors(i,:),  onesFacesNum,1), 'filled');
    hold on;
end
xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');
title('LDA Projection to 3D Space');



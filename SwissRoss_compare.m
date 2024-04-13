clc;
clear;
[X,tt] = SwissRoll();
figure(2)
scatter3(X(1,:),X(2,:),X(3,:), 20,tt,'filled');
train_data=X;

mean_data = mean(train_data,2);% 计算样本均值
centered_data = train_data - mean_data;% 中心化数据
cov_matrix = centered_data*centered_data';% 计算协方差矩阵
[eigenvectors, eigenvalues] = eig(cov_matrix);% 计算协方差矩阵的特征值和特征向量
[~, idx] = sort(diag(eigenvalues), 'descend');
eigenvectors=eigenvectors(:,idx);%特征向量按特征值排序

Y=eigenvectors(:,1:2)'*X;
figure;
scatter(Y(1,:),Y(2,:), 20,tt,'filled');

%LPP降维%
k = 5; % KNN中的近邻数
dist_matrix = pdist2(X', X');% 使用pdist2函数计算距离矩阵
[~, nearest_indices] = mink(dist_matrix, k+1, 2); % 计算近邻关系，加1是因为每个样本的最近邻是它自己
% 构建权重矩阵
sigma = 1; % 高斯核函数的参数
rbf_dist = exp(-(dist_matrix.^2 / (2*sigma^2))); % 高斯核函数(径向基函数)计算权重
for i = 1:size(X, 2)
    neighbors = nearest_indices(i, 2:end);% 获取第i个样本的近邻索引
    W(i, neighbors) = rbf_dist(i,neighbors);% 更新权重矩阵
    W(neighbors,i) = rbf_dist(neighbors,i);% 权重矩阵是对称的
end
D = diag(sum(W, 2));% 计算度矩阵
L = D - W;% 计算拉普拉斯矩阵
[eigenvectors, eigenvalues] = eig(pinv(X*D*X') * X*L*X');% 计算广义特征值问题的解
% eigenvalues(eigenvalues < 0.1) = NaN;
[~, idx] = sort(diag(eigenvalues));
eigenvectors=eigenvectors(:,idx);%特征向量按特征值升序排列

Y=eigenvectors(:,1:2)'*X;
% 投影到二维
figure;
scatter(Y(1,:),Y(2,:), 20,tt,'filled');


Y = isomap(X',5, 2);

% 可视化结果
figure;
scatter(Y(:, 1), Y(:, 2), 20,tt,'filled');

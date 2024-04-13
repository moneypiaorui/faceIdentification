function [eigenvectors] = LPP(X,sigma)
% 输入参数 训练集(按列放置)，高斯核函数的sigma
% 返回值  排好序后的特征向量(按列放置)

%LPP降维%
k = 10; % KNN中的近邻数
dist_matrix = pdist2(X', X');% 使用pdist2函数计算距离矩阵(欧氏距离的平方)
[~, nearest_indices] = mink(dist_matrix, k+1, 2); % 计算近邻关系，加1是因为每个样本的最近邻是它自己
% 构建权重矩阵
rbf_dist = exp(-(dist_matrix./ (2*sigma^2))); % 高斯核函数(径向基函数)计算权重
W=zeros(size(X,2),size(X,2));
for i = 1:size(X, 2)
    neighbors = nearest_indices(i, 2:end);% 获取第i个样本的近邻索引
    W(i, neighbors) = rbf_dist(i,neighbors);% 更新权重矩阵
    W(neighbors,i) = rbf_dist(neighbors,i);% 权重矩阵是对称的
end
D = diag(sum(W, 2));% 计算度矩阵
L = D - W;% 计算拉普拉斯矩阵
[eigenvectors, eigenvalues] = eig(pinv(X*D*X') * X*L*X');% 计算广义特征值问题的解
eigenvalues(eigenvalues < 0.1) = NaN;
[~, idx] = sort(diag(eigenvalues));
eigenvectors=eigenvectors(:,idx);%特征向量按特征值升序排列

end
function [eigenvectors] = PCA(train_data)
% 输入参数 训练集(按列放置)
% 返回值  排好序后的特征向量(按列放置)

%pca降维%
mean_data = mean(train_data,2);% 计算样本均值
centered_data = train_data - mean_data;% 中心化数据
cov_matrix = centered_data*centered_data';% 计算协方差矩阵
[eigenvectors, eigenvalues] = eig(cov_matrix);% 计算协方差矩阵的特征值和特征向量
[~, idx] = sort(diag(eigenvalues), 'descend');
eigenvectors=eigenvectors(:,idx);%特征向量按特征值排序

end
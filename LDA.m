function [eigenvectors] = LDA(train_data,train_labels, num_classes)
% 输入参数 训练集(按列放置)，训练集标签(行向量)，类别数
% 返回值  排好序后的特征向量(按列放置)

% LDA降维 %
% 计算每个类别的均值向量
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
end
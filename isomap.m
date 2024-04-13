function [Y, D] = isomap(X, k, d)
    % X: 输入数据，每一行表示一个样本
    % k: 邻居数
    % d: 目标低维空间的维度
    
    % 计算样本之间的欧氏距离
    D = pdist2(X, X);
    
    % 找到每个样本的k个最近邻
    [~, indices] = mink(D, k+1, 2);
    
    % 构建邻接矩阵
    A = zeros(size(X, 1));
    for i = 1:size(X, 1)
        A(i, indices(i, 2:end)) = 1;
        A(indices(i, 2:end), i) = 1;
    end
    
    % 使用Graph类计算最短路径距离
    G = graph(A);
    D = distances(G);
    
    % 使用MDS算法将距离矩阵映射到d维空间
    mds = cmdscale(D);
    
    % 取前d个维度作为最终结果
    Y = mds(:, 1:d);
end
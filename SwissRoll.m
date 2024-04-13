% 生成瑞士卷数据集
function [X,tt] = SwissRoll()
N = 2000;
rand('state',123456789);
%Gaussian noise
noise = 0.001*randn(1,N);
%standard swiss roll data
tt = (3*pi/2)*(1+2*rand(1,N));
height = 21*rand(1,N);
X = [(tt+ noise).*cos(tt); height; (tt+ noise).*sin(tt)];
end

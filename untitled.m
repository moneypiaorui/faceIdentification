clear;



% 1.人脸数据集的导入与数据处理（400张图，一共40人，一人10张）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reshaped_faces=[];
for i=1:40    
    for j=1:10       
        if(i<10)
           a=imread(strcat('E:\Project\matlab\face\database\AR_Gray_50by40\AR00',num2str(i),'-',num2str(j),'.tif'));     
        else
            a=imread(strcat('E:\Project\matlab\face\database\AR_Gray_50by40\AR0',num2str(i),'-',num2str(j),'.tif'));  
        end          
        b = reshape(a,2000,1); %将每一张人脸拉成列向量
        b=double(b); %utf-8转换为double类型，避免人脸展示时全灰的影响       
        reshaped_faces=[reshaped_faces, b];  
    end
end
 
% 取出前30%作为测试数据，剩下70%作为训练数据
test_data_index = [];
train_data_index = [];

train_data=[];
test_data=[];

for i=0:39
    test_data_index = [test_data_index 10*i+1:10*i+3];
    train_data_index = [train_data_index 10*i+4:10*(i+1)];
end

train_data = reshaped_faces(:,train_data_index);
test_data = reshaped_faces(:, test_data_index);

% 2.图像求均值，中心化
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% 求平均脸
mean_face = mean(train_data,2);
%waitfor(show_face(mean_face)); %平均脸展示，测试用
 
% 中心化
centered_face = (train_data - mean_face);
%用于展示中心化后某些训练图片 测试用
waitfor(show_faces(centered_face));
% 3.求协方差矩阵、特征值与特征向量并排序
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% 协方差矩阵
cov_matrix = centered_face * centered_face';
[eigen_vectors, dianogol_matrix] = eig(cov_matrix);
% 从对角矩阵获取特征值
eigen_values = diag(dianogol_matrix);
% 对特征值按索引进行从大到小排序
[sorted_eigen_values, index] = sort(eigen_values, 'descend'); 
% 获取排序后的征值对应的特征向量
sorted_eigen_vectors = eigen_vectors(:, index);
% 特征脸(所有）
all_eigen_faces = sorted_eigen_vectors;
%用于展示某些特征脸 测试用
waitfor(show_faces(all_eigen_faces));

%%人脸重构
 
% 取出第一个人的人脸，用于重构
single_face = centered_face(:,1);
 
index = 1;
for dimensionality=20:20:160
 
    % 取出相应数量特征脸（前n大的特征向量，用于重构人脸）
    eigen_faces = all_eigen_faces(:,1:dimensionality);
 
    % 重建人脸并显示
        rebuild_face = eigen_faces * (eigen_faces' * single_face) + mean_face;
        subplot(2, 4, index); %两行四列
        index = index + 1;
        fig = show_face(rebuild_face);
        title(sprintf("dimensionality=%d", dimensionality));    
end
waitfor(fig);
% 5.人脸识别
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
index = 1;       
Y = [];
% KNN
for i=10:10:160
 
    for k=1:6
    % 取出相应数量特征脸
   eigen_faces = all_eigen_faces(:,1:i);
    % 测试、训练数据降维
    projected_train_data = eigen_faces' * (train_data - mean_face);
    projected_test_data = eigen_faces' * (test_data - mean_face);
        % 用于保存最小的k个值的矩阵
        % 用于保存最小k个值对应的人标签的矩阵
        minimun_k_values = zeros(size(projected_train_data,2),1);
        label_of_minimun_k_values = zeros(size(projected_train_data,2),1);
 
        % 测试脸的数量
        test_face_number = size(projected_test_data, 2);
 
        % 识别正确数量
        correct_predict_number = 0;
 
        % 遍历每一个待测试人脸
        for each_test_face_index = 1:test_face_number
 
            each_test_face = projected_test_data(:,each_test_face_index);
 
            for each_train_face_index = 1:size(projected_train_data,2)
                minimun_k_values(each_train_face_index,1) = norm(each_test_face - projected_train_data(:,each_train_face_index));
                label_of_minimun_k_values(each_train_face_index,1) = floor((train_data_index(1,each_train_face_index) - 1) / 10) + 1;
            end
 
            % 找出k个值中最大值及其下标
            [minimun_k_values, I] = sort(minimun_k_values);
            label_of_minimun_k_values=label_of_minimun_k_values(I);
 
            
 
            % 最终得到距离最小的k个值以及对应的标签
            % 取出出现次数最多的值，为预测的人脸标签
            predict_label = mode(label_of_minimun_k_values(1:k,1));
            real_label = floor((test_data_index(1,each_test_face_index) - 1) / 10)+1;
 
            if (predict_label == real_label)
                %fprintf("预测值：%d，实际值:%d，正确\n",predict_label,real_label);
                correct_predict_number = correct_predict_number + 1;
            else
                %fprintf("预测值：%d，实际值:%d，错误\n",predict_label,real_label);
            end
        end
        % 单次识别率
        correct_rate = correct_predict_number/test_face_number;
 
        Y = [Y correct_rate];
 
        fprintf("k=%d，i=%d，总测试样本：%d，正确数:%d，正确率：%1f\n", k, i,test_face_number,correct_predict_number,correct_rate);
    end
end
% 求不同k值不同维度下的人脸识别率及平均识别率
Y = reshape(Y,k,16);
waitfor(waterfall(Y));
avg_correct_rate=mean(Y);
waitfor(plot(avg_correct_rate));



%内用函数定义
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 输入向量，显示脸
function fig = show_face(vector)
    fig = imshow(uint8(reshape(vector, [50, 40])),[]);
end
 
% 显示矩阵中某些脸
function fig = show_faces(eigen_vectors)
    count = 1;
    index_of_image_to_show = [1,5,10,15,20,30,50,70,100,150];
    for i=index_of_image_to_show
        subplot(2,5,count);
        fig = show_face(eigen_vectors(:, i));
        title(sprintf("i=%d", i));
        count = count + 1;
    end
end

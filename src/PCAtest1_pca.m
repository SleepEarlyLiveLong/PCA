% 测试 PCA 对数据的降维作用和主成分分析效果
% PCAtest1_pca.m:
%   This file is used for testing the PCA(Principle Component
%   Analysis) algorithm, I give 2 examples here and thay are
%   Speech Gender Recognition and Digital Handwriting Recognition.
%
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% 用例一 语音性别识别
close;clear;
load('resource\voive_data.mat');
data = v_d(:,1:20);
data = data';
num_all = size(data,2);
num_male = num_all/2;
target_dimension = 20;
Res = myPCA(data,target_dimension);                  % for all
% Res = myPCA(data(:,1:num_male),target_dimension);    % for male
% Res = myPCA(data(:,num_male+1:num_all),target_dimension);  % for female

% 语音性别识别，三维可视化
component_one = 1;
component_two = 2;
component_three = 3;
figure(1);
scatter3(Res.Y(component_one,1:num_male),Res.Y(component_two,1:num_male),Res.Y(component_three,1:num_male),'b');   % male
hold on;
scatter3(Res.Y(component_one,num_male+1:num_all),Res.Y(component_two,num_male+1:num_all),Res.Y(component_three,num_male+1:num_all),'r');  % female
legend('Male','Female');
title('语音性别识别-主成分分析-三维可视化');
xlabel(['第',num2str(component_one),'主成分']);
ylabel(['第',num2str(component_two),'主成分']);
zlabel(['第',num2str(component_three),'主成分']);

% 语音性别识别，二维可视化
component_one = 1;
component_two = 2;
figure(2);
scatter(Res.Y(component_one,1:1584),Res.Y(component_two,1:1584),'b');
hold on;
scatter(Res.Y(component_one,1585:3168),Res.Y(component_two,1585:3168),'r');
legend('Male','Female');
title('语音性别识别-主成分分析-二维可视化');
xlabel(['第',num2str(component_one),'主成分']);
ylabel(['第',num2str(component_two),'主成分']);

% 语音性别识别，PCA累计贡献率
consum = zeros(target_dimension,1);
for i=1:target_dimension
    consum(i) = sum(Res.contrb(1:i));
end

% 语音性别识别，数据汇总
% save('resource\genderRes.mat','genderRes');

% ----------------------- 作图 -----------------------
load('resource\genderRes.mat');
figure(3);
x=1:target_dimension;
yyaxis left;
bar([genderRes.MF_contrbt_single,genderRes.M_contrbt_single,genderRes.F_contrbt_single]);
ylabel('Contribution rate of principal components');ylim([0 0.6]);
yyaxis right;
plot(x,genderRes.MF_contrbt_total,'-*g',x,genderRes.M_contrbt_total,'-*b',x,genderRes.F_contrbt_total,'-*r');
hold on;
plot(x,0.85*ones(1,target_dimension),'--');
ylabel('Cumulative contribution rate of principal components');ylim([0 1.08]);
title(['Contribution Rate Analysis of PCA (Gender) (target_dimension=',num2str(target_dimension),')']);
xlabel('The first k principal component');
xlim([0 target_dimension]);
legend('In total','Male only','Female only',...
    'In total','Male only','Female only','Orientation','horizontal');

%% 语音性别识别 Res.Y后处理作为其他用途
voice_dedimention = (Res.Y)';
addi = [zeros(num_male,1);ones(num_male,1)];
voice_dedimention = [voice_dedimention,addi];
save('resource\voice_dedimention.mat','voice_dedimention');

%% 用例二 手写体MNIST识别
load('resource\DIGITS.mat');
% ------------get mnist train data
mnist_dtrain = DIGITS{11}.Data_train;
for i=12:20
    mnist_dtrain = [mnist_dtrain;DIGITS{i}.Data_train];
end
mnist_dtrain = mnist_dtrain';
% ------------get mnist test dat
mnist_dtest = DIGITS{1}.Data_test;
for i=2:10
    mnist_dtest = [mnist_dtest;DIGITS{i}.Data_test];
end
mnist_dtest = mnist_dtest';
mnist_dall = [mnist_dtrain,mnist_dtest];
% ------------do the PCA analysis
target_dimension = 80;
Res = myPCA(mnist_dall,target_dimension);

% MNIST手写体识别，三维可视化
component_one = 1;
component_two = 2;
component_three = 3;
figure(4);
% train_num = [980 1135 1032 1010 982 892 958 1028 974 1009];
train_num = [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949];
scatter3(Res.Y(component_one,1:train_num(1)),Res.Y(component_two,1:train_num(1)),Res.Y(component_three,1:train_num(1)),'MarkerEdgeColor',[0 1 0]);   % male
hold on;
scatter3(Res.Y(component_one, 1+sum(train_num(1:1)):sum(train_num(1:2))),Res.Y(component_two, 1+sum(train_num(1:1)):sum(train_num(1:2))),Res.Y(component_three, 1+sum(train_num(1:1)):sum(train_num(1:2))),'MarkerEdgeColor',[1 0 0]);
scatter3(Res.Y(component_one, 1+sum(train_num(1:2)):sum(train_num(1:3))),Res.Y(component_two, 1+sum(train_num(1:2)):sum(train_num(1:3))),Res.Y(component_three, 1+sum(train_num(1:2)):sum(train_num(1:3))),'MarkerEdgeColor',[0 1 0]);
scatter3(Res.Y(component_one, 1+sum(train_num(1:3)):sum(train_num(1:4))),Res.Y(component_two, 1+sum(train_num(1:3)):sum(train_num(1:4))),Res.Y(component_three, 1+sum(train_num(1:3)):sum(train_num(1:4))),'MarkerEdgeColor',[0 0 1]);
scatter3(Res.Y(component_one, 1+sum(train_num(1:4)):sum(train_num(1:5))),Res.Y(component_two, 1+sum(train_num(1:4)):sum(train_num(1:5))),Res.Y(component_three, 1+sum(train_num(1:4)):sum(train_num(1:5))),'MarkerEdgeColor',[1 1 0]);
scatter3(Res.Y(component_one, 1+sum(train_num(1:5)):sum(train_num(1:6))),Res.Y(component_two, 1+sum(train_num(1:5)):sum(train_num(1:6))),Res.Y(component_three, 1+sum(train_num(1:5)):sum(train_num(1:6))),'MarkerEdgeColor',[1 0 1]);
scatter3(Res.Y(component_one, 1+sum(train_num(1:6)):sum(train_num(1:7))),Res.Y(component_two, 1+sum(train_num(1:6)):sum(train_num(1:7))),Res.Y(component_three, 1+sum(train_num(1:6)):sum(train_num(1:7))),'MarkerEdgeColor',[0 1 1]);
scatter3(Res.Y(component_one, 1+sum(train_num(1:7)):sum(train_num(1:8))),Res.Y(component_two, 1+sum(train_num(1:7)):sum(train_num(1:8))),Res.Y(component_three, 1+sum(train_num(1:7)):sum(train_num(1:8))),'MarkerEdgeColor',[0.5 0.5 0.5]);
scatter3(Res.Y(component_one, 1+sum(train_num(1:8)):sum(train_num(1:9))),Res.Y(component_two, 1+sum(train_num(1:8)):sum(train_num(1:9))),Res.Y(component_three, 1+sum(train_num(1:8)):sum(train_num(1:9))),'MarkerEdgeColor',[0 0.5 0.5]);
scatter3(Res.Y(component_one, 1+sum(train_num(1:9)):sum(train_num(1:10))),Res.Y(component_two, 1+sum(train_num(1:9)):sum(train_num(1:10))),Res.Y(component_three, 1+sum(train_num(1:9)):sum(train_num(1:10))),'MarkerEdgeColor',[0.5 0 0]);
legend('0','1','2','3','4','5','6','7','8','9');
title('MNIST(训练集)手写体识别-主成分分析-三维可视化');
xlabel(['第',num2str(component_one),'主成分']);
ylabel(['第',num2str(component_two),'主成分']);
zlabel(['第',num2str(component_three),'主成分']);

% mnist，PCA累计贡献率
consum = zeros(target_dimension,1);
for i=1:target_dimension
    consum(i) = sum(Res.contrb(1:i));
end

% ----------------------- 作图 -----------------------
figure(5);
x=1:target_dimension;
yyaxis left;
bar(Res.contrb(:,1));
ylabel('Contribution rate of principal components');ylim([0 0.12]);
yyaxis right;
plot(x,consum,'-*g');
hold on;
plot(x,0.85*ones(1,target_dimension),'--');
ylabel('Cumulative contribution rate of principal components');ylim([0 1.08]);
title(['Contribution Rate Analysis of PCA (MNIST) (target_dimension=',num2str(target_dimension),')']);
xlabel('The first k principal component');
xlim([0 target_dimension+1]);
%% 手写体MNIST识别 Res.Y后处理作为其他用途
delete = [1:4,25:30,54:57];
Res.Y(delete,:) = [];
DIGITS_dedimention = DIGITS;
temp = zeros(20,1);
datalen = zeros(20,1);
for i=1:10
    temp(i) = size(DIGITS{i,1}.Data_test,1);
    temp(10+i) = size(DIGITS{i+10,1}.Data_train,1);
end
for i=1:20
    datalen(i) = sum(temp(1:i));
end
datalen = [0;datalen(1:20)];
for i=1:10
    DIGITS_dedimention{i,1}.Data_test = (Res.Y(:,datalen(i)+1:datalen(i+1)))';
    DIGITS_dedimention{i+10,1}.Data_train = (Res.Y(:,datalen(i+10)+1:datalen(i+11)))';
end
save('resource\DIGITS_dedimention.mat','DIGITS_dedimention');
%%
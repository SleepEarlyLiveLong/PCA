% 测试 PCA 对数据的变换作用—— PCA 的实质是数据以质心为中心集体旋转，不改变相对位置
% PCAtest2_rotate.m:
%   This file is used for testing the PCA(Principle Component
%   Analysis) algorithm, Here is the result ——
%   The essence of PCA is data revolving collectively around the center
%   of mass without changing their relative position.
%
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% 2-D space
close;clear;
mu=[10,20];                       %数学期望
sigma=[1 4;4 20];            %协方差矩阵
r = mvnrnd(mu,sigma,200);      %生成50个样本
figure;plot(r(:,1),r(:,2),'r*');hold on;
X = r';
Res = myPCA(X,2);
P = Res.P;
Y = Res.Y;
center = Res.center;
plot(Y(1,:),Y(2,:),'b*');
plot(center(1),center(2),'yo','MarkerFaceColor','y');
axis equal;
myarrow(center,10*P(1,:),'g');
myarrow(center,10*P(2,:),'b');
myarrow([0 0],10*[1 0],'g');
myarrow([0 0],10*[0 1],'b');

%% 3-D space
% 数学期望
mu=[10,10,10];                       
% 协方差矩阵需要是对称的半正定矩阵
N = 3;
% x = diag(rand(N,1));
x = diag([1 3 30]);
u = orth(rand(N,N));
sigma = u' * x * u;
r = mvnrnd(mu,sigma,1000);      %生成若干个样本
figure;plot3(r(:,1),r(:,2),r(:,3),'r*');hold on;
grid on;
X = r';
Res = myPCA(X,N);
P = Res.P;
Y = Res.Y;
center = Res.center;
plot3(Y(1,:),Y(2,:),Y(3,:),'b*');
plot3(center(1),center(2),center(3),'yo','MarkerFaceColor','y');
axis equal;
myarrow(center,20*P(1,:),'k');
myarrow(center,20*P(2,:),'g');
myarrow(center,20*P(3,:),'b');
myarrow([0 0 0],20*[1 0 0],'k');
myarrow([0 0 0],20*[0 1 0],'g');
myarrow([0 0 0],20*[0 0 1],'b');
%% 
% 测试 PCA 在图像压缩方面的应用
% 还存在一个疑点: 为得到压缩图像需要保留的数据量居然比原图的数据量还大
% PCAtest3_imgcprs.m:
%   This file is used for testing the PCA(Principle Component
%   Analysis) algorithm used in the image compression field.
%
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% read image
close;clear;
im = imread('resource\standard_lena.bmp');
figure;subplot(1,2,1);imshow(im,[]);title('原图');
im = double(im);
[m,n] = size(im);
% 图像拆分
len = 16;
PicMtx = im2col(im, [len len], 'distinct');
% 每块图像去均值
% pic_m = ones(size(PicMtx,1),1)*mean(PicMtx);
% PicMtx = PicMtx - pic_m;
% 进行PCA运算
k = 48;
rho = (len^2+512*512/(len^2))*k/(512*512);
Res = myPCA(PicMtx,k);
P = Res.P;
Y = Res.Y;

% 图像恢复 只要得到 P、Y 两个矩阵就可以恢复出原图
Xr = (P)'*Y + mean(PicMtx,2)*ones(1,m*n/(len^2));  %按字段加回均值
% Xr = Xr + pic_m;  %再按图像块加回均值
s = col2im(Xr, [len len], [m,n], 'distinct');
subplot(1,2,2);imshow(s,[]);title(['PCA压缩: n=',num2str(len^2),', k=',num2str(k),', rho=',num2str(rho)]);
% 数据压缩率(保留的数据/原图像素点数
% 这样看来必然有 rou>1 那还压缩什么呢
rou = (262144/(len^2)+k)*len^2/262144;

%% 小结
% 若每块图像去均值，则主成分贡献率：
% [0.420191084255480;0.157584199103716;0.102331321400645;0.0712510794015086;
%     0.0360761582954789;0.0269531551095093;0.0237799035728113;0.0211749242997639;
%     0.0190269846507898;0.0137355375670231]......
% 否则，主成分贡献率：
% [0.878608530162519;0.0512988260957464;0.0192322284956799;0.0119552882721241;
%     0.00861821662411581;0.00439999801493981;0.00329010432954140;0.00288337185857526;
%     0.00258161170165692;0.00229630181231791]......
% 然鹅，图像压缩恢复的效果都差不多
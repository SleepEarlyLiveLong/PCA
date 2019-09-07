% ���� PCA ��ͼ��ѹ�������Ӧ��
% ������һ���ɵ�: Ϊ�õ�ѹ��ͼ����Ҫ��������������Ȼ��ԭͼ������������
% PCAtest3_imgcprs.m:
%   This file is used for testing the PCA(Principle Component
%   Analysis) algorithm used in the image compression field.
%
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% read image
close;clear;
im = imread('resource\standard_lena.bmp');
figure;subplot(1,2,1);imshow(im,[]);title('ԭͼ');
im = double(im);
[m,n] = size(im);
% ͼ����
len = 16;
PicMtx = im2col(im, [len len], 'distinct');
% ÿ��ͼ��ȥ��ֵ
% pic_m = ones(size(PicMtx,1),1)*mean(PicMtx);
% PicMtx = PicMtx - pic_m;
% ����PCA����
k = 48;
rho = (len^2+512*512/(len^2))*k/(512*512);
Res = myPCA(PicMtx,k);
P = Res.P;
Y = Res.Y;

% ͼ��ָ� ֻҪ�õ� P��Y ��������Ϳ��Իָ���ԭͼ
Xr = (P)'*Y + mean(PicMtx,2)*ones(1,m*n/(len^2));  %���ֶμӻؾ�ֵ
% Xr = Xr + pic_m;  %�ٰ�ͼ���ӻؾ�ֵ
s = col2im(Xr, [len len], [m,n], 'distinct');
subplot(1,2,2);imshow(s,[]);title(['PCAѹ��: n=',num2str(len^2),', k=',num2str(k),', rho=',num2str(rho)]);
% ����ѹ����(����������/ԭͼ���ص���
% ����������Ȼ�� rou>1 �ǻ�ѹ��ʲô��
rou = (262144/(len^2)+k)*len^2/262144;

%% С��
% ��ÿ��ͼ��ȥ��ֵ�������ɷֹ����ʣ�
% [0.420191084255480;0.157584199103716;0.102331321400645;0.0712510794015086;
%     0.0360761582954789;0.0269531551095093;0.0237799035728113;0.0211749242997639;
%     0.0190269846507898;0.0137355375670231]......
% �������ɷֹ����ʣ�
% [0.878608530162519;0.0512988260957464;0.0192322284956799;0.0119552882721241;
%     0.00861821662411581;0.00439999801493981;0.00329010432954140;0.00288337185857526;
%     0.00258161170165692;0.00229630181231791]......
% Ȼ�죬ͼ��ѹ���ָ���Ч�������
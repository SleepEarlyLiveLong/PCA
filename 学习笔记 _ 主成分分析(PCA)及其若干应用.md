
# <center><font face="宋体"> 学习笔记|主成分分析(PCA)及其若干应用 </font></center>

*<center><font face="Times New Roman" size = 3> Author：[chentianyang](https://github.com/chentianyangWHU) &emsp;&emsp; E-mail：tychen@whu.edu.cn &emsp;&emsp; [Link](https://github.com/chentianyangWHU/PCA)</center>*

**概要：** <font face="宋体" size = 3> 前段时间学习了一些矩阵分解算法，包括主成分分析(Principal Component Analysis, PCA)、独立成分分析(Independent Component Analysis, ICA)、非负矩阵分解(Non-negative Matrix Factorization, NMF)等。接下来我将用三五篇博客的篇幅简要介绍这些方法，以及他们的若干项应用。这篇博客先介绍主成分分析。</font>

**关键字：** <font face="宋体" size = 3 >矩阵分解; 主成分分析; PCA</font>

# <font face="宋体"> 1 背景说明 </font>

&emsp;&emsp; <font face="宋体">近来在学机器学习，果然是对数学要求很高。尤其是矩阵方面需要有丰富的知识储备，否则不论是算法还是代码都是很难看懂的，更别说作出什么东西来了。矩阵分解是矩阵分析中一个很重要的话题，以此为出发点可以发散出许许多多的算法，而且这些算法在实际工程项目中往往具有非常不错的效果和广阔的应用。主成分分析(PCA)在形式上可以归为矩阵分解问题一类，但实际上却是作为一种数据分析算法“闻名天下”的。顾名思义，PCA就是将一组数据的“主要成分”提取出来而忽略剩下的次要内容，达到数据降维的效果，以减少运算资源消耗或达成其他目的。从哲学上讲，这一想法体现了马克思主义哲学中唯物辩证法的核心——矛盾的观点，即抓主要矛盾。当我们用哲学的思想指导实践时，往往能对现实作出某称程度上的预判。因此，我们应当相信PCA算法存在的必然性和重要性，同时这也有助于我们更加迅速深入地了解PCA的本质和核心。</font>

# <font face="宋体"> 2 算法原理 </font>

## <font face="宋体"> 2.1 PCA简介</font>

&emsp;&emsp; <font face="宋体">大概主成分分析是最重要的降维方法之一。在数据压缩、消除冗余和数据噪音消除等领域都有广泛的应用，一般我们提到降维最容易想到的算法就是它。PCA试图用数据最主要的若干方面来代替原有的数据，这些最主要的方面首先需要保证蕴含了原始数据中的大量信息，其次需要保证相互之间不相关。因为相关代表了数据在某种程度上的“重叠”，也就相当于冗余性没有清除干净。现在，问题转变为什么样的特性能代表原始数据中的“信息”，怎样才能“最大程度地”保留信息。让我们先看图1。</font>

<table>
   <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/201812281520012.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70"  width="90%"> <font face="Times New Roman" size = 2> &ensp;&ensp;&ensp;&ensp;&ensp;图1 数据投影（1） </font></center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20181228152049888.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70"  width="90%"><font face="Times New Roman" size = 2>    &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;图1 数据投影（2） </font></center></td>
	</tr>
</table>

&emsp;&emsp; <font face="宋体">上面的截图来源于[白马负金羁](https://blog.csdn.net/baimafujinji/article/details/79376378)，实际上是个动图，我这里截图截成了静止的图像。上图都是将二维空间(平面)中的一群离散点投影到一维空间(直线)上去，那么应该怎样投才能使一维空间上保留尽可能多的原始信息？直观来看，显然是图1_(1)的效果比图1_(2)好，因为在(2)中点都纠集到了一团，很多点在直线上的投影重合了，这样最多只能保留一个点的信息，而图(1)中的投影相对更加离散。</font>

&emsp;&emsp; <font face="宋体">说到离散，我就想到了西游记中的唐僧一别太宗十七载，去国离乡取西经的故事。明年，中外合拍的电影西游记即将正式开机，章老师将继续扮演美猴王孙悟空……咳咳，打住打住。说到离散，我就想起了数学上的方差概念，数据越离散保留的信息就越多，反之亦反。到这里，PCA的另一个指导思想就呼之欲出了：方差即信息。</font>

## <font face="宋体"> 2.2 基本原理</font>
&emsp;&emsp; <font face="宋体">因为我比较懒，而且花时间抄书确实没有什么意思，所以公式什么的就不列举了。这里介绍几个把PCA的数学原理写得非常好的博客，水平比我不知道高到哪里去了，请大家移步观赏：</font>

&emsp;&emsp; <font face="宋体">1. [PCA的数学原理](http://blog.codinglabs.org/articles/pca-tutorial.html)：从简单的2维数据开始讲起，深入浅出、娓娓道来，绝对是教材中的上品，零基础学生的福音；</font>

&emsp;&emsp; <font face="宋体">2. [Principal Component Analysis](http://setosa.io/ev/principal-component-analysis/)：PCA动图展示，支持用户交互，有时候不得不承认，人家老外做的东西就是比国内做得好，而且不止一点半点；</font>

&emsp;&emsp; <font face="宋体">3. [PCA(主成分分析)](https://yoyoyohamapi.gitbooks.io/mit-ml/content/%E7%89%B9%E5%BE%81%E9%99%8D%E7%BB%B4/articles/PCA.html)：这是斯坦福机器学习课程(吴恩达授课)笔记系列的一部分，它最大的价值是针对成机器学习课程形成了一个完整的系列，参考价值很高；</font>

&emsp;&emsp; <font face="宋体">4. [機器/統計學習:主成分分析](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8-%E7%B5%B1%E8%A8%88%E5%AD%B8%E7%BF%92-%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90-principle-component-analysis-pca-58229cd26e71)：看上去应该是台湾人做的，示例详实、图文并茂，也很好；</font>

&emsp;&emsp; <font face="宋体">5. [主成分分析(PCA)与Kernel PCA](https://blog.csdn.net/baimafujinji/article/details/79376378)：我上文提到的白马负金羁的博客，公式不建议深究，KPCA也不完整，但是那张动图确实很好；</font>

&emsp;&emsp; <font face="宋体">6. [PCA主成分分析学习总结](https://zhuanlan.zhihu.com/p/32412043)：这是知乎上的一篇学习总结，从基础知识入手，中规中矩，质量较高，而且这个知乎专栏的内容质量也较高。</font>

&emsp;&emsp; <font face="宋体">部分是英文资料，部分资料需要翻墙，这也是没办法的事情。</font>

## <font face="宋体"> 2.3 形式化表达</font>

&emsp;&emsp; <font face="宋体">将一个问题通过数学方法表达出来，就是形式化表达，这是求解问题的第一步。由于本人在CSDN上打公式不熟练，所以就用PPT代劳了，如图2所示。</font>

<center><img src="https://img-blog.csdnimg.cn/20181228161022433.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70" width="80%">  </center><center><font face="宋体" size=2 > 图2 PCA算法的形式化表示 </font> </center>

&nbsp;
&emsp;&emsp; <font face="宋体">这就是PCA的形式化表示，对于一个输入的$n$行$m$列矩阵，PCA的目标就是将它降至$k$维，$k<n$，输出$k$行$m$列矩阵，注意数据量$m$是保持不变的。</font>

# <font face="宋体"> 3 算法步骤与代码 </font>

&emsp;&emsp; <font face="宋体">经过中间一系列计算步骤(这里不一一展现)，最后得到了PCA算法的实现步骤如图3所示(仍然是PPT代劳)：</font>

<center><img src="https://img-blog.csdnimg.cn/20181228161645585.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70" width="80%">  </center><center><font face="宋体" size=2 > 图3 PCA算法的形式化表示 </font> </center>

&nbsp;
&emsp;&emsp; <font face="宋体">有了步骤，接下来就只剩小学生都会做的编程了。</font>

&emsp;&emsp; <font face="宋体">MATLAB代码如下：</font>
```
function Res = myPCA(X,R,varargin)
%MYPCM - The Principal Component Analysis(PCA) function.
%   To calculate the result after PCA, one kind of 
%   dimension-reduction technical of characteristics.
%   Here are some useful reference material:
%   https://www.jianshu.com/p/2fad63faa773
%   http://blog.codinglabs.org/articles/pca-tutorial.html
%
%   Res = myPCA(X,R)
%   Res = myPCA(X,R,DIM)
% 
%   Input - 
%   X: a N*M matrix containing M datas with N dimensions;
%   R: number of dimensions to be reduced to, normally, R<N;
%   DIM: specifies a dimension DIM to arrange X.
%       DIM = 1: X(N*M)
%       DIM = 2: X(M*N)
%       DIM = otherwisw: error
%   Output - 
%   Res  : the result of PCA of the input matrix X;
%       Res.P: a R*N matrix containing R bases with N dimensions;
%       Res.Y: a R*M matrix containing M datas with R dimensions;
%       Res.contrb: a R*1 vector containing the contribution rate 
%                   of each principal component.
%       Res.sumcontrb: a scaler means the sum contribution rate
%                   of all principal components.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% parameter test
% parameter number test
narginchk(2,3);
narg = numel(varargin);
DIM = [];
switch narg
    case 0
    case 1
        DIM = varargin{:};
    otherwise
        error('Error! Input parameter error.');
end
if isempty(DIM)
    DIM = 1;
end

% parameter correction test
if ~ismatrix(X) || ~isreal(R)
    error('Error! Input parameters error.');
end
[N,M] = size(X);
if R > N
    error('Error! The 2nd parameter should be smaller than the col. of the 1st one.');
elseif R == N
    warning('Warning! There is no dimension-reduction effect.');
end
if DIM == 2
    X = X';
elseif DIM~=1 && DIM~=2
    error('Error! Parameter DIM should be either 1 or 2.');
end

%% core algorithm
center = mean(X,2);
X = X - repmat(center,1,M);      % zero_centered for each LINE/Field
C = X*(X')/M;                       % the Covariance matrix of X, C(N*N)
[eigenvector,eigenvalue] = eig(C);
[B,order] = sort(diag(eigenvalue),'descend');
% calculate contribution-rate matrix
contrb = zeros(R,2);
contrb(:,1) = B(1:R)/sum(B);             
for i=1:R
    contrb(i,2) = sum(contrb(1:i,1));
end
P = zeros(R,N);
% convert eigenvectors from columns to lines.
for i=1:R
    P(i,:) = eigenvector(:,order(i))';
end
Y = P*X;

%% get result
Res.contrb = contrb;
Res.P = P;
Res.Y = Y;
Res.center = center;

end
```

# <font face="宋体"> 4 PCA实例与应用 </font>

&emsp;&emsp; <font face="宋体">接下来我将就上面的PCA算法做一些test code，来检测以下PCA的性能，顺便也能加深理解。</font>

## <font face="宋体"> 4.1 PCA的实质</font>

&emsp;&emsp; <font face="宋体">我认为，PCA的实质是数据的**整体旋转**。</font>

&emsp;&emsp; <font face="宋体">之前说到，PCA算法旨在寻找某些相互独立的方向，使得原始数据在这些方向上的投影尽可能离散。拆开来理解就是这么一回事：</font>

&emsp;&emsp; <font face="宋体">1. 尽可能离散就是需要尽可能多地保留原始信息；</font>
&emsp;&emsp; <font face="宋体">2. 寻找的方向要相互独立是为了避免保留下来的信息存在冗余；</font>

&emsp;&emsp; <font face="宋体">3. “若干个”应当少于、最多等于原始数据维数，否则就不是降维了(注意，从数学上来讲是可以做到升维的，这个与Kernel Trick，即核方法有关，本系列的后续文章会介绍)。</font>

&emsp;&emsp; <font face="宋体">还记得文章开头的那2张截图吗，原来的动图是一条轴线以数据质心为中心旋转，从旋转簇中选一条作为投影方向。**从相对运动的角度来说，这等价于坐标轴不动，所有数据整体旋转，在不改变数据之间相对位置的情况下，使得自己方差最大的方向从大到小地对准坐标系中一组基底的方向。**</font>

&emsp;&emsp; <font face="宋体">相关的实验结果如图4所示：</font>

<table>
   <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/20181228164816264.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70"  width="90%"> <font face="Times New Roman" size = 2> &ensp;&ensp;&ensp;&ensp;&ensp;图4(1) 2维数据PCA变换前后 </font></center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20181228164859623.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70"  width="90%"><font face="Times New Roman" size = 2>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;图4(2) 3维数据PCA变换前后 </font></center></td>
	</tr>
</table>

&emsp;&emsp; <font face="宋体">如上面两张图所示，红色为原数据，蓝色为PCA变换后没有降维的数据。图4(1)中方差由大到小的**2**个不相关轴分别是绿色和蓝色的，图4(2)中方差由大到小的**3**个不相关轴分别是黑色、绿色和蓝色的。可以看见变换后数据质心移到了原点，且根据数据离散程度分别与原坐标系的基底$i,j,k$对齐，数据集的形态没有改变(数据点的相对位置不变)。在此基础上，如果要降维，自然是根据数据在各个轴上的方差值由小到大舍弃舍弃某一维即可。因此，不妨对PCA的原理做如下描述：</font>

&emsp;&emsp; <font face="宋体">**将数据整体旋转后，根据数据在各维度上投影的方差值由小到大地舍弃维度。**</font>

## <font face="宋体"> 4.2 用PCA降维</font>

&emsp;&emsp; <font face="宋体">这是PCA最直接的应用，降低了维度就可以减小运算数据量。</font>

### <font face="宋体"> 4.2.1 语音性别识别数据集</font>

&emsp;&emsp; <font face="宋体">现在用PCA降低[语音性别识别实验](https://blog.csdn.net/ctyqy2015301200079/article/details/83346310)中所用数据集的维度，该数据集原来的数据维度为20，我在之前的博客中有相关说明。降维的效果如图5所示。</font>

<center><img src="https://img-blog.csdnimg.cn/20181228171451119.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70" width="100%">  </center><center><font face="宋体" size=2 > 图5 语音数据集主成分分析 </font> </center>

&nbsp;
&emsp;&emsp; <font face="宋体">图中蓝、绿、黄三色分别代表总计、男性、女性，柱状图为各主成分的信息贡献率，折线图为前若干主成分的合计信息贡献率。可以发现主成分贡献率逐渐减小而越前面的主成分贡献率越大。一般而言我们需要保留原数据85%以上的信息，在本例中前5个主成分的累计贡献率就达到这一数据，因此理论上可以直接将20维数据降到5维，数据量减少75%而信息量只减少15%。</font>

### <font face="宋体"> 4.2.1 MNIST数据集</font>

&emsp;&emsp; <font face="宋体">总计20维可能还太小，PCA的效果难以表现出来。现在对MNIST数据集分析，该数据集我也在之前的[博客](https://blog.csdn.net/ctyqy2015301200079/article/details/83380533)中谈到，每一个手写体数字是28*28的矩阵，共计784维。降维的效果如图6所示。</font>

<center><img src="https://img-blog.csdnimg.cn/20181228172241462.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70" width="100%">  </center><center><font face="宋体" size=2 > 图6 MNIST数据集主成分分析 </font> </center>

&nbsp;
&emsp;&emsp; <font face="宋体">图中仅展现了前80维，因为后面的维度基本没有信息。在本例中约前60个主成分的累计贡献率就达到85%，因此理论上可以直接将784维数据降到60+维，数据量减少90%以上而信息量只减少15%。</font>

## <font face="宋体"> 4.3 用PCA做数据可视化</font>

&emsp;&emsp; <font face="宋体">这一条相当于是4.2的特例，将高维数据降低到2维或是3维，放到坐标系中自然就可视了。图7所示是语音数据集的可视化。</font>

<table>
   <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/20181228185640464.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70"  width="90%"> <font face="Times New Roman" size = 2> &ensp;&ensp;&ensp;&ensp;&ensp;图7(1) PCA变换后为2维数据 </font></center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20181228173710681.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70"  width="90%"><font face="Times New Roman" size = 2>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;图7(2) PCA变换后为3维数据 </font></center></td>
	</tr>
</table>

&nbsp;
&emsp;&emsp; <font face="宋体">图7分别为原始数据(20维)被降维到2维和3维后的图像，蓝色是男性数据，红色是女性数据。另一方面，如果降维后两者区分较明显，通过这种方法就可以直接进行分类。</font>

## <font face="宋体"> 4.4 用PCA做图像压缩</font>

&emsp;&emsp; <font face="宋体">用PCA做图像压缩的基本思路是图2所示的公式：</font>

$$P_{k \times n}\cdot X_{n \times m}=Y_{k \times m}$$

&emsp;&emsp; <font face="宋体">PCA的思路是设矩阵$X_{n \times m}$表示一整一张$m$行$n$列的图像，通过矩阵$P_{k \times n}$将其降至$k$维，得到$k$行$m$列矩阵$Y_{k \times m}$。那么反过来，只要知道和$Y_{k \times m}$和$P_{k \times n}$就能计算出矩阵$X_{n \times m}$，而矩阵$P_{k \times n}$和$Y_{k \times m}$的数据量是可以小于矩阵$X_{n \times m}$的。因此，定义保留的数据量与原图像数据量之比为压缩率，则PCA做图像压缩的压缩率为：</font>

$$\rho = \frac{k(m+n)}{mn}$$

&emsp;&emsp; <font face="宋体">在实际处理中，往往是将原图像无重叠拆分成$m$个子图、每个子图拉成一长度为$n$的列向量来拼成原矩阵$X_{n \times m}$的。现不妨设一图像长宽都是512像素，则有：</font>

$$mn=512^2$$

&emsp;&emsp; <font face="宋体">因此，</font>

$$\rho = \frac{k(m+n)}{mn} \geqslant \frac{2k \sqrt{mn}}{mn} \geqslant \frac{2k}{\sqrt{mn}}=\frac{k}{256}$$

&emsp;&emsp; <font face="宋体">若将这张图拆分为8*8的小图，那么$n=64,m=4096$，根据PCA原理，需满足$k \leqslant n$，不妨令$k=32$，此时原图信息仍能保留绝大部分，而压缩率为50.78%，相当于数据量减少一半而信息几乎不丢失。图8所示为用PCA做图像压缩的效果，原图是经典的$lena$图像，是512在\*512的灰度图。</font>

<center><img src="https://img-blog.csdnimg.cn/20181228193904735.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2N0eXF5MjAxNTMwMTIwMDA3OQ==,size_16,color_FFFFFF,t_70" width="100%">  </center><center><font face="宋体" size=2 > 图8 用PCA做图像压缩示例 </font> </center>

&nbsp;
&emsp;&emsp; <font face="宋体">上面的图像压缩率是25.391%。</font>

&emsp;&emsp; <font face="宋体">从压缩公式可知，要想进一步压缩，有2条途径可以走：要么减小$k$，要么使得$m$和$n$数值尽可能接近，从而使柯西不等式尽可能接近等号的取值。而要在减小$k$的同时不止于损失过多的数据，就需要$n$尽可能小，结果就是$m$和$n$的数值差距越来越大，从而偏离柯西不等式的最小值，同时还需记得参与运算的都是整数，且要将图像完整分拆，这些都需要考虑。所以说这是一个 **trade off** 的过程，细节部分还需要具体问题具体分析。</font>

&emsp;&emsp; <font face="宋体">另外，若不能将大图完整拆分，也可以考虑给大图补上白边，或是有重叠地拆分，或者所拆分得到的小图也不一定非得是正方形，等等方法都可以尝试。</font>

# <font face="宋体"> 5 小结 </font>

&emsp;&emsp; <font face="宋体">本文初步探讨了主成分分析算法(PCA)的原理以及若干应用，下面做一个小结。</font>

&emsp;&emsp; <font face="宋体">PCA是一种著名的数据**降维算法**，它应用的条件是数据/特征之间具有明显的**线性相关性**，它的两个主要的指导思想是**抓主要矛盾**和**方差即信息**，它的基本应用是**数据降维**，以此为基础还有数据可视化、数据压缩存储、异常检测、特征匹配与距离计算等。从数学上理解，它是一种**矩阵分解算法**；从物理意义上理解，它是**线性空间上的线性变换**。</font>

&emsp;&emsp; <font face="宋体">PCA的核心代码已经在文中给出，其他的都是一些测试代码，这里就不给出了。关于PCA，还有KPCA(Kernel PCA，核PCA)、MPCA(Multilinear PCA，多线性PCA)等变形，可以有效弥补标准PCA算法存在的某种缺陷，关于这些内容，我会在这个系列后面的文章中简要介绍。</font>

&emsp;&emsp; <font face="宋体">本文为原创文章，转载或引用务必注明来源及作者。</font>
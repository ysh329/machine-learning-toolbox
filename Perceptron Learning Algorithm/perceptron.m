function [w, cur_best_w, mis_class, cur_mis_class] = perceptron(X, t, maxiter)
%% The perceptron learning algorithm AND pocket algorithm
%   Explanation:
%   其实这里应该是PLA和 Pocket 算法的综合
%   cur_best_w：返回的是目前最好的权重系数（即基于贪心策略的口袋算法）
%   w：返回的是PLA最后一次迭代
%   w：若迭代次数会达到迭代最大次数，即数据无法线性可分或者数据可以线性可分但是还未到
%   w：若迭代次数在最大次数内，即数据可以线性可分
%
%   Args:
%   X : D*N维输入数据 【所有样本的特征值】
%   【X的列数】：样本个数
%   【X的行数】：（单个）样本的特征数（维度）
%   t ： {+1,-1}【真实类标签】元素个数即X的列数（样本个数）
%
%   Return:
%   w ： [w0 w1 w2]【模型参数值】 w of PLA
%   cur_best_w ：口袋算法当前最好的w, w of PA
%   mis_class ： 错误分类数据点数

%% 变量检查
% unique 对矩阵或向量的元素去重（如果参数是行向量，unique 结果是行向量，如果参数是列向量或矩阵，则器结果是列向量）
% numel(x)：返回x向量或矩阵的元素个数
if numel(unique(t)) ~= 2
    return
elseif max(t)~=1
    return
elseif min(t)~=-1
    return
end

%% 参数初始化
% dim(X的行数)：特征矩阵X的单个样本的【特征个数】
% num_data(X的列数)：特征矩阵X的【样本个数】
[dim, num_data] = size(X);

% w(3, 1)：模型的（权重）系数，加的1行是截距项
w = ones(dim+1,1);%如果是两个特征（w0是截距项对应的权重参数）：w = [w0 w1 w2]'

% 在特征矩阵X之上扩充一行（元素初始值全部为1）
% X(3, num_data)，X是特征矩阵，在原本两个特征x1(用来拟合w1),x2(用来拟合w2)的基础上，又增加了一行（x0特征，用来拟合模型参数w0）
X = [ones(1,num_data);X];
% 最大迭代次数 input parameter
%maxiter = 100000;
% 当前迭代分错的样本点个数
mis_class = 0;
% 当前迭代次数
iter = 0;
% 当前错误最少时候的权重（口袋算法参数初始化）
cur_mis_class = size(X, 2);
cur_best_w = w;

%% 算法迭代（学习的过程 = 模型参数改变的过程）
while iter < maxiter
    iter = iter + 1;
    % w'(1,3) * X(3, num_data)  ===>  y(1, num_data)
    % y(1, num_data)：按照当前的模型参数w(包括w0, w1, w2)预测出来的num_data个样本的【预测值】
    y = w' * X;

    % label(1, num_data)：初始化预测标签
    label = ones(1, num_data);%{+1,-1}
    label(y <= 0) = -1;
    % 通过预测类标签label与真实类标签t比较，找出分错的点
    % t(1, num_data)：t是真实类标签的值，向量t中元素值为1或-1
    % index(1, num_data)：元素值是1（预测正确）或0（预测错误）
    index = find(label~=t); %错误分类的点
    % numel返回矩阵（index）的元素个数（即分错的个数）
    mis_class = numel(index); %错误分类点的数目
    
    
    
    %% 口袋算法（贪心策略）的选择
    if mis_class <= cur_mis_class
        %disp(iter);
        cur_iter = iter;
        cur_best_w = w;
        cur_mis_class = mis_class;
        cur_index = index;
    end    
    
    % 若分错个数为0，迭代完成，否则继续迭代（先调整参数w，即学习过程）
    if mis_class==0
        break
    end
     
    % 对mis_class个分错点进行参数w(w0,w1,w2)的修正（学习）
    % i：第i个分错点，用i遍历所有错误点索引下标index
    for i = 1:mis_class
        % index：错误点的索引下标，用i遍历所有错误点索引下标index
        % t(index(i))：第index(i)个样本（该样本是分错的样本，因为index都是分错样本点的索引下标）的【真实类标签】
        % X(:,index(i))：第index(i)个样本（该样本是分错的样本，因为index都是分错样本点的索引下标）的【样本特征】（x0,x1,x2）
        w = w +  t(index(i)) * X(:,index(i));
    end
end
%% 结果输出
if iter == maxiter
    disp(['达到最大迭代次数' num2str(maxiter)])
end
disp('======================')
disp(['当前迭代次数iter:' num2str(iter)]);
disp('======================')
disp(['PLA模型参数w:' num2str(w')]);
disp(['PLA分错点个数mis_class:' num2str(numel(index))]);
disp(['PLA分错点的下标:' num2str(index)]);
disp('======================')
disp(['当前迭代次数iter:' num2str(cur_iter)]);
disp(['PA模型参数w:' num2str(cur_best_w')]);
disp(['PA分错点个数mis_class:' num2str(cur_mis_class)]);
disp(['PA分错点的下标:' num2str(cur_index)]);
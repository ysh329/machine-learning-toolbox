%% perceptron.m 的例子（可视化数据部分只能跑二维特征的数据）
%% 目前的问题：在还未找到可完美分类的线之前的PLA，其决策边界
%%            可视化出来的错误分类点，难以解释，或者说就是错的。
%%            但PA是基于PLA的，似乎又不可能出错。
%%
%% 若计算出的直线（后面可视化结果）没有将数据切分开（有分错的点），
%% 两个可能的原因：
%% 1.数据原本就不是线性可分的
%% 2.数据虽然是线性可分的，但迭代的次数还不够

%% Step0.initial workspace and etc.
clear; clf; clc;

%% Step1.Prepare data, 这里的数据是二维的
X = [-17, -13, 16, 12, 7, 15, 14, 17, 31, 47, 21, 40, 32, 27, 51, 22;
    131, 171, -42, 41, 51, -120, 214, 48, 67, 54, 78, 53, 32, 12, 121, 132];
t = ones(1, size(X, 2));
t(:, 1:size(X, 2)/2) = -t(:, 1:size(X, 2)/2);
maxiter = 100000;

%% Step2.Plot scatter plot
figure(1);
plot(X(1, 1:size(X, 2)/2)', X(2, 1:size(X, 2)/2)', 'ro',...
    X(1, size(X, 2)/2+1:size(X, 2))', X(2, size(X, 2)/2+1:size(X, 2))', 'bo');
title('Realistic Class Using PLA');

%% Step3.Calculate using perceptron
[w, cur_best_w] = perceptron(X, t, maxiter);

%% Step4.plot decision boundary based on PLA
%% 绘制决策边界，不过这里是绘制的是二维的
% x0*w0 + x1*w1 + x2*w2 = 0
% x2*w2 = -(x0*w0 + x1*w1)
% x2 = -(x0*w0 + x1*w1)/w2
X = [ones(1, size(X, 2));
    X];
decision_X2 = -(X(1,:).*w(1) + X(2,:).*w(2)) ./ w(3);
hold on;
plot(X(2, :), decision_X2, 'b');
plot(X(2, 1:size(X, 2)/2), decision_X2(1:size(X, 2)/2), 'magentao');
plot(X(2, size(X, 2)/2+1:size(X, 2)), decision_X2(size(X, 2)/2+1:size(X, 2)), 'greeno');
grid on;
legend('class 1(-)', 'class 2(+)',...
    'parameter vector W',...
    'class1(-) on decision boundary', 'class2(+) on decision boundary');
xlabel('feature x1');
ylabel('feature x2');

%% Step5.plot decision boundary based on PA
%% 绘制决策边界，不过这里是绘制的是二维的
w = cur_best_w;
figure(2);
plot(X(2, 1:size(X, 2)/2)', X(3, 1:size(X, 2)/2)', 'ro',...
    X(2, size(X, 2)/2+1:size(X, 2))', X(3, size(X, 2)/2+1:size(X, 2))', 'bo');
title('Realistic Class Using PA');
decision_X2 = -(X(1,:).*w(1) + X(2,:).*w(2)) ./ w(3);
hold on;
plot(X(2, :), decision_X2, 'b');
plot(X(2, 1:size(X, 2)/2), decision_X2(1:size(X, 2)/2), 'magentao');
plot(X(2, size(X, 2)/2+1:size(X, 2)), decision_X2(size(X, 2)/2+1:size(X, 2)), 'greeno');
grid on;
legend('class 1(-)', 'class 2(+)',...
    'parameter vector W',...
    'class1(-) on decision boundary', 'class2(+) on decision boundary');
xlabel('feature x1');
ylabel('feature x2');
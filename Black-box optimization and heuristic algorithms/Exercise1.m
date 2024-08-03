%%%%%%%%%%%%%% 粒子群算法求函数极值 %%%%%%%%%%%%%
% 初始化参数
clear all; % 清除工作空间的所有变量
close all; % 关闭所有图形窗口
clc; % 清除命令窗口的内容

% 设置算法参数
N = 100; % 粒子数量
D = 2; % 搜索空间的维数
T = 200; % 最大迭代次数
c1 = 1.5; % 个体学习因子
c2 = 1.5; % 社会学习因子
Wmax = 0.8; % 惯性权重的最大值
Wmin = 0.4; % 惯性权重的最小值
Xmax = 4; % 粒子位置的最大值
Xmin = -4; % 粒子位置的最小值
Vmax = 1; % 粒子速度的最大值
Vmin = -1; % 粒子速度的最小值

% 初始化粒子的位置和速度
x = rand(N,D) * (Xmax-Xmin)+Xmin; % 随机初始化粒子位置
v = rand(N,D) * (Vmax-Vmin)+Vmin; % 随机初始化粒子速度

% 初始化粒子个体的最优位置和最优值
p = x; % 粒子个体的最优位置初始化为粒子的初始位置
pbest = ones(N,1); % 初始化粒子个体的最优适应度值
for i = 1:N
    pbest(i) = func2(x(i,:)); % 计算初始位置的适应度值
end

% 初始化全局最优位置和最优值
g = ones(1,D); % 初始化全局最优位置
gbest = inf; % 初始化全局最优适应度值
for i = 1:N
    if (pbest(i) < gbest) % 更新全局最优值和位置
        g = p(i,:);
        gbest = pbest(i);
    end
end

% 记录每一代的全局最优适应度值
gb = ones(1,T);

% 迭代直到达到最大迭代次数或精度要求
for i = 1:T
    for j = 1:N
        % 更新粒子个体的最优位置和最优值
        if (func2(x(j,:)) < pbest(j))
            p(j,:) = x(j,:); % 更新个体最优位置
            pbest(j) = func2(x(j,:)); % 更新个体最优适应度值
        end
        
        % 更新全局最优位置和最优值
        if (pbest(j) < gbest)
            g = p(j,:); % 更新全局最优位置
            gbest = pbest(j); % 更新全局最优适应度值
        end
        
        % 计算动态惯性权重
        w = Wmax - (Wmax - Wmin) * i / T;
        
        % 更新粒子的位置和速度
        v(j,:) = w * v(j,:) + c1 * rand * (p(j,:) - x(j,:)) + c2 * rand * (g - x(j,:));
        x(j,:) = x(j,:) + v(j,:);
        
        % 边界条件处理，确保粒子位置和速度在预设的范围内
        for ii = 1:D
            if (v(j,ii) > Vmax) | (v(j,ii) < Vmin)
                v(j,ii) = rand * (Vmax - Vmin) + Vmin; % 速度超出范围时重置速度
            end
            if (x(j,ii) > Xmax) | (x(j,ii) < Xmin)
                x(j,ii) = rand * (Xmax - Xmin) + Xmin; % 位置超出范围时重置位置
            end
        end
    end
    % 记录当前迭代的全局最优适应度值
    gb(i) = gbest;
end

% 输出最优个体的位置和最优适应度值
disp(g); % 显示最优个体的位置
disp(gb(end)); % 显示最优适应度值

% 绘制适应度进化曲线
figure; % 创建新图形窗口
plot(gb); % 绘制适应度值随迭代次数的变化
xlabel('迭代次数'); % x轴标签
ylabel('适应度值'); % y轴标签
title('适应度进化曲线'); % 图形标题

% 适应度函数，定义了要优化的目标函数
function value = func2(x)
    value = 3 * cos(x(1) * x(2)) + x(1) + x(2)^2; % 目标函数表达式
end
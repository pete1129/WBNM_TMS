% function scatter_plot(X, Y)
% 
% n = length(X);
% index=find(tril(ones(n),-1));
% p=polyfit(X(index), Y(index),1);
% y = polyval(p,X(index));
% % y=p(1,1)*X(index)+p(1,2);
% [r,p]=corr(X(index), Y(index));
% r=roundn(r,-3);
% if p>0.01&&p<0.05
%     p='p < 0.05';
% elseif p>0.001&&p<0.01
%     p='p < 0.01';
% elseif p<0.001
%     p='p < 0.001';
% end
% 
% hold on
% scatter(X(index),Y(index),5)
% % plot(X(index),Y(index),'.k')
% plot(X(index),y,'r')
% color='red';
% fontsize=12;
% fontname='Times New Roman';
% text('string',['r = ' num2str(r)],'Units','normalized','position',[0.75,0.2],'FontSize',fontsize,'FontName',fontname,'Color',color)
% text('string',p,'Units','normalized','position',[0.75,0.1],'FontSize',fontsize,'FontName',fontname,'Color',color)
% hold off
% box on
% end

% function scatter_plot(Y, X)
%     % 交换 X 和 Y 位置
%     n = length(X);
%     index = find(tril(ones(n), -1));
% 
%     % 线性回归
%     p = polyfit(Y(index), X(index), 1);
%     y_fit = polyval(p, Y(index));
% 
%     % 计算相关性
%     [r, p_val] = corr(Y(index), X(index));
%     r = round(r, 3); % 保留三位小数
% 
%     % p 值格式化
%     if p_val > 0.01 && p_val < 0.05
%         p_text = 'p < 0.05';
%     elseif p_val > 0.001 && p_val < 0.01
%         p_text = 'p < 0.01';
%     elseif p_val < 0.001
%         p_text = 'p < 0.001';
%     else
%         p_text = ['p = ' num2str(p_val, '%.3g')];
%     end
% 
%     % ---- 创建图形窗口 ----
%     figure;
% 
%     % ---- 主图（散点图 + 二维核密度） ----
%     ax1 = axes('Position', [0.15, 0.15, 0.6, 0.6]); % 主图位置
%     hold on;
% 
%     % 计算二维核密度估计
%     [xi, yi] = meshgrid(linspace(0, 1, 100), linspace(0, 4, 100));
%     density = ksdensity([Y(index), X(index)], [xi(:), yi(:)]);
%     density = reshape(density, size(xi));
% 
%     % 核密度背景
%     contourf(xi, yi, density, 10, 'LineStyle', 'none');
%     colormap(sky); % 使用 Sky 颜色
%     colorbar('Position', [0.82, 0.15, 0.02, 0.6]); % 右侧 colorbar
%     alpha(0.6);
% 
%     % 绘制散点图和拟合线
%     scatter(Y(index), X(index), 5);
%     plot(Y(index), y_fit, 'r');
% 
%     % 相关性文本
%     text(0.75, 0.2, ['r = ' num2str(r)], 'Units', 'normalized', 'FontSize', 12, 'FontName', 'Times New Roman', 'Color', 'red');
%     text(0.75, 0.1, p_text, 'Units', 'normalized', 'FontSize', 12, 'FontName', 'Times New Roman', 'Color', 'red');
% 
%     hold off;
%     box on;
% 
%     % ---- X 轴边际密度图（上方） ----
%     ax2 = axes('Position', [0.15, 0.75, 0.6, 0.15]); % 细长的X轴密度曲线
%     hold on;
%     [f, xi] = ksdensity(Y(index));
%     plot(xi, f, 'r', 'LineWidth', 1.5);
%     hold off;
%     xlim([0 1]); % 保证 X 轴范围正确
% 
%     ax2.YAxis.TickDirection = 'out';
% 
%     % ---- Y 轴边际密度图（主图和 colorbar 之间） ----
%     ax3 = axes('Position', [0.77, 0.15, 0.05, 0.6]); % 紧挨 colorbar
%     hold on;
%     [f, yi] = ksdensity(X(index));
%     plot(f, yi, 'r', 'LineWidth', 1.5);
%     hold off;
%     xlim([0 4]); % 保证 Y 轴范围正确
% 
%     ax3.XAxis.TickDirection = 'out';
% 
%     % 确保所有图层可见
%     linkaxes([ax1, ax2], 'x');
%     linkaxes([ax1, ax3], 'y');
% end


function scatter_plot(Y, X) % 交换X和Y的位置

n = length(X);
index = find(tril(ones(n), -1));
p = polyfit(Y(index), X(index), 1); % 交换X和Y的位置
y = polyval(p, Y(index));
% y = p(1,1) * Y(index) + p(1,2);
[r, p] = corr(Y(index), X(index));
r = roundn(r, -3);
if p > 0.01 && p < 0.05
    p = 'p < 0.05';
elseif p > 0.001 && p < 0.01
    p = 'p < 0.01';
elseif p < 0.001
    p = 'p < 0.001';
end

hold on
scatter(Y(index), X(index), 5) % 交换X和Y的位置
plot(Y(index), y, 'r') % 交换X和Y的位置
color = 'red';
fontsize = 12;
fontname = 'Times New Roman';
text('string', ['r = ' num2str(r)], 'Units', 'normalized', 'position', [0.75, 0.2], 'FontSize', fontsize, 'FontName', fontname, 'Color', color)
text('string', p, 'Units', 'normalized', 'position', [0.75, 0.1], 'FontSize', fontsize, 'FontName', fontname, 'Color', color)
hold off
box on
end

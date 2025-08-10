function bar_plot(y,a,b)
x = categorical({a,b});
x = reordercats(x,{a,b});
b=bar(x,y,0.4);
b.FaceColor = 'flat';
b.CData(2,:)=[0.85 0.325 0.098];
xtips1 = b.XData;
ytips1 = b.YData;
labels1 = string(b.YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
end


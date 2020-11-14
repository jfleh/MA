pkg load statistics

x = [-3:.01:3];
y = 0.45*normpdf(x,-1,0.2)+0.55*normpdf(x,1,0.2);
plot(x,y,'LineWidth',2.0,"color","k")
line ([0.1 0.1], [0 0.2], "linewidth", 2, "linestyle", "-", "color", "b");
line ([1 1], [0 0.2], "linewidth", 2,"linestyle", "-", "color", "r");
set(gca,'XTick',-3:1:3,"fontsize",20);
set(gca,'YTick',0:0.6:1.2,"fontsize",20);
print -dpng CMleqMAP

x = [0:.1:400];
y_gam = exppdf(x,100);
plot(x,y_gam,'LineWidth',2.0,"color","k")
line ([100 100], [0 0.01/7],"linewidth", 2, "linestyle", "-", "color", "b");
line ([0 0], [0 0.01/7],"linewidth", 2, "linestyle", "-", "color", "r");
set(gca,'XTick',0:100:400,"fontsize",20);
set(gca,'YTick',0:0.005:0.01,"fontsize",20);
print -dpng CMgeqMAP
pkg load statistics

x = [-10:.1:50];
y_prior = gampdf(x,1.5,5);
y_noise = normpdf(x,4,2);
plot(x,y_prior,'LineWidth',2.0,"color","k")
hold on
plot(x,y_noise,'LineWidth',2.0,"color","k")
set(gca,'XTick',-3:1:3,"fontsize",20);
set(gca,'YTick',0:0.6:1.2,"fontsize",20);
y_posterior =0;

%print -dpng CMleqMAP
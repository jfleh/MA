x=[-2:0.1:2];
y_relu=max(0,x);
y_tanh=tanh(x);
y_sigmoid=1./(1+exp(-x));

plot(x,y_relu,'LineWidth',2.5,"color","k")
xlim([-2 2])
ylim([-2 2])
set(gca,'XTick',-2:1:2,"fontsize",25);
set(gca,'YTick',-2:1:2,"fontsize",25);
print -dpng ReLU

plot(x,y_tanh,'LineWidth',2.5,"color","k")
xlim([-2 2])
ylim([-2 2])
set(gca,'XTick',-2:1:2,"fontsize",25);
set(gca,'YTick',-2:1:2,"fontsize",25);
print -dpng Tanh

plot(x,y_sigmoid,'LineWidth',2.5,"color","k")
xlim([-2 2])
ylim([-2 2])
set(gca,'XTick',-2:1:2,"fontsize",25);
set(gca,'YTick',-2:1:2,"fontsize",25);
print -dpng Sigmoid

clear 
x1=[-2:0.1:0];
x2=[0:0.1:2];
y_elu1=exp(x1)-1;
y_elu2=x2;

plot(x1,y_elu1,'LineWidth',2.5,"color","k")
hold on
plot(x2,y_elu2,'LineWidth',2.5,"color","k")
xlim([-2 2])
ylim([-2 2])
set(gca,'XTick',-2:1:2,"fontsize",25);
set(gca,'YTick',-2:1:2,"fontsize",25);
print -dpng ELU
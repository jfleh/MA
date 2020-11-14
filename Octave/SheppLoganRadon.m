pkg load image

P = phantom('Modified Shepp-Logan',500);
Seite = -250:250;
imagesc(Seite,Seite,P);
title('f(x,y)',"fontsize",20);
xlabel('x',"fontsize",20);
ylabel('y',"fontsize",20);
set(gca,'XTick',-250:100:250,"fontsize",20);
set(gca,'YTick',-250:100:250,"fontsize",20);
colormap(gray);
colorbar("fontsize",20)
#imshow(P)
print -dpng SheppLogan

theta = 0:180;
[R,xp] = radon(P,theta);
imagesc(theta,xp,R);
title('Rf(r,\phi)',"fontsize",20);
xlabel('\phi (degrees)',"fontsize",20);
ylabel('r',"fontsize",20);
set(gca,'XTick',0:30:180,"fontsize",20);
set(gca,'YTick',-250:100:250,"fontsize",20);
colormap(gray);
colorbar("fontsize",20)
print -dpng SheppLoganRadon
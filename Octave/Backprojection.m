pkg load image

n = 2^8;                 % size of mask
M = zeros(n);
I = 1:n; 
x = I-n/2;                % mask x-coordinates 
y = n/2-I;                % mask y-coordinates
[X,Y] = meshgrid(x,y);    % create 2-D mask grid
R1 = 2^5;                   % aperture radius
A = (X.^2 + Y.^2 <= R1^2); % circular aperture of radius R
M(A) = 1;                

Theta = 0:90:179;
[R,xp] = radon(M,Theta);
Rinv = iradon(R,[],'linear','none')
imagesc(Rinv)
set(gca,'visible','off')
colormap(gray)
print -dpng Backprojection2Angles

Theta = 0:45:179;
[R,xp] = radon(M,Theta);
Rinv = iradon(R,[],'linear','none')
imagesc(Rinv)
set(gca,'visible','off')
colormap(gray)
print -dpng Backprojection4Angles

Theta = 0:30:179;
[R,xp] = radon(M,Theta);
Rinv = iradon(R,[],'linear','none')
imagesc(Rinv)
set(gca,'visible','off')
colormap(gray)
print -dpng Backprojection6Angles

Theta = 0:15:179;
[R,xp] = radon(M,Theta);
Rinv = iradon(R,[],'linear','none')
imagesc(Rinv)
set(gca,'visible','off')
colormap(gray)
print -dpng Backprojection12Angles

Theta = 0:10:179;
[R,xp] = radon(M,Theta);
Rinv = iradon(R,[],'linear','none')
imagesc(Rinv)
set(gca,'visible','off')
colormap(gray)
print -dpng Backprojection18Angles

Theta = 0:5:179;
[R,xp] = radon(M,Theta);
Rinv = iradon(R,[],'linear','none')
imagesc(Rinv)
set(gca,'visible','off')
colormap(gray)
print -dpng Backprojection36Angles

Theta = 0:1:179;
[R,xp] = radon(M,Theta);
Rinv = iradon(R,[],'linear','none')
imagesc(Rinv)
set(gca,'visible','off')
colormap(gray)
print -dpng Backprojection180Angles
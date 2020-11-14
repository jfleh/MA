pkg load image

% Create a logical image of a circle with specified
% diameter, center, and image size.
% First create the image.
imageSizeX = 500;
imageSizeY = 500;
[columnsInImage rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
% Next create the circle in the image.
centerX = 250;
centerY = 250;
radius = 250;
circlePixels = (rowsInImage - centerY).^2 ...
    + (columnsInImage - centerX).^2 <= radius.^2;
% circlePixels is a 2D "logical" array.

I = zeros(500,500);
I(circlePixels) = 1;

centerX = 250;
centerY = 250;
radius = 200;
circlePixels = (rowsInImage - centerY).^2 ...
    + (columnsInImage - centerX).^2 <= radius.^2;

I(circlePixels) = 0;   

#imshow(I)
Seite = -250:250;
imagesc(Seite,Seite,I);
title('f(x,y)',"fontsize",20);
xlabel('x',"fontsize",20);
ylabel('y',"fontsize",20);
set(gca,'XTick',-250:100:250,"fontsize",20);
set(gca,'YTick',-250:100:250,"fontsize",20);
colormap(gray);
colorbar("fontsize",20)
print -dpng Ring

theta = 0:180;
[R,xp] = radon(I,theta);
imagesc(theta,xp,R);
title('Rf(r,\phi)',"fontsize",20);
xlabel('\phi (degrees)',"fontsize",20);
ylabel('r',"fontsize",20);
set(gca,'XTick',0:30:180,"fontsize",20);
set(gca,'YTick',-250:100:250,"fontsize",20);
colormap(gray);
colorbar("fontsize",20)
print -dpng RingRadon
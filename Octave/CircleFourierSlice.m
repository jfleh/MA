pkg load image

n = 2^9;                 % size of mask
K = zeros(n);
I = 1:n; 
x = I-n/2;                % mask x-coordinates 
y = n/2-I;                % mask y-coordinates
[X,Y] = meshgrid(x,y);    % create 2-D mask grid
R1 = 2^7;                   % aperture radius
A = (X.^2 + Y.^2 <= R1^2); % circular aperture of radius R
K(A) = 1;                 % set mask elements inside aperture to 1

imagesc(K)
set(gca,'visible','off')
colormap(flipud(gray))
print -dpng CircleNew

n = 2^9;                 % size of mask
M = zeros(n);
I = 1:n; 
x = I-n/2;                % mask x-coordinates 
y = n/2-I;                % mask y-coordinates
[X,Y] = meshgrid(x,y);    % create 2-D mask grid
R1 = 10;                   % aperture radius
A = (X.^2 + Y.^2 <= R1^2); % circular aperture of radius R
M(A) = 1;                

[R,xp] = radon(M,90);
figure
x=-20:0.1:20;
y=2*sqrt(100-x.^2);
plot(x,y,'LineWidth',2.0,"color","k")
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
print -dpng CircleRadonSliceNew

R_fft = (1/sqrt(2*pi)).*abs(fftshift(fft(R)));
tt = xp(165:565,1);
test = R_fft(1:401,1);
plot(-200:200,transpose(R_fft(165:565,1)),'LineWidth',2.0,"color","k");
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
print -dpng CircleRadonSliceFourierNew

DP = fftshift(fft2(M));
imagesc((1/2*pi).*abs(DP(129:384,129:384)))
set(gca,'visible','off')
colormap(flipud(gray))
print -dpng CircleFourierNew
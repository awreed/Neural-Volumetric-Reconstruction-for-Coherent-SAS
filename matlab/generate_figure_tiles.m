close all;
clear all;
volumeViewer close;

font_size = 11;



%names = {''}
font_size = 11;
alpha_scale = 0.8;

img_resize = 1;

intensity = [0 20 40 120 220 1024];
alpha = alpha_scale * [0 0 0.15 0.3 0.38 1.];

queryPoints = linspace(min(intensity),max(intensity),256);
alphamap = interp1(intensity,alpha,queryPoints)';

blue = [0 0 0; 67.8/3 84.7/3 90.2/2; 67.8/2 84.7/2 90.2/1; 2*67.8/1 2*84.7/1 2*120/1; 255 255 255; 255 255 255]/255;
red = [0 0 0; 96.9/3 79.2/3 78.8/2; 96.9/2 79.2/2 78.8/1; 2*120/1 2*79.2/1 2*78.8/1; 255 255 255; 255 255 255]/255;
colormap_blue = interp1(intensity,blue,queryPoints);
colormap_red = interp1(intensity,red,queryPoints);

data = load(['./data.mat']);
data = data.scene;


data = double(abs(data));
data = preprocess(data, 0.2, 1);


render_x = 800;
render_y = 800;


vol = volshow(data,Colormap=colormap_blue,Alphamap=alphamap);
viewer = vol.Parent;

hFig = viewer.Parent;
hFig.Units = 'inches';
hFig.Position=[0., 0., 14, 14];
viewer.Units = 'inches';
viewer.Position = [0., 0., 14, 14];


viewer.BackgroundColor = [1., 1., 1.];
viewer.BackgroundGradient = 'off';

sz = size(data);

dist = sqrt(sz(1)^2 + sz(2)^2 + sz(3)^2);
%dist = dist - 200;
center = sz/2 - 0.5;

theta_vector = 230;
%theta_vector = 180;

%theta_vector = ones(12)*210;
viewer.Units = 'pixels';
viewer.OrientationAxes = 'off';
hFig.Units = 'pixels';

theta = deg2rad(theta_vector);
pos = center + ([cos(theta), sin(theta), ones(size(theta))*0.5]*dist);
viewer.CameraPosition = pos;
viewer.CameraZoom = 1.5;

drawnow;

img = getframe(hFig);
%img.cdata = imresize(img.cdata, [render_x, render_y]);

save_name = strcat("./matlab_render.png");
imwrite(img.cdata, save_name);

volumeViewer close;


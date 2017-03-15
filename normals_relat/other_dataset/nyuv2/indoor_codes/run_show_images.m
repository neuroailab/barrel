% Visualizes all of the images.
addpath('common/');
Consts;

for ii = ii : consts.numImages
  
  % Load the RGB image.
  load(sprintf(consts.imageRgbFilename, ii), 'imgRgb');
  
  % Load the Depth image.
  load(sprintf(consts.imageDepthFilename, ii), 'imgDepth');
  
  % Load the Labels.
  imgRegions = get_regions(ii);
  
  
  sfigure(1);
  subplot(1,3,1);
  imagesc(imgRgb);
  title('RGB');
  axis off;
  axis equal;
  
  subplot(1,3,2);
  imagesc(imgDepth);
  title('Depth');
  axis off;
  axis equal;
  
  subplot(1,3,3);
  vals = randperm(max(imgRegions(:)));
  vis_regions(imgRegions, vals);
  title('Instances');
  axis off;
  axis equal;
  
  set(gcf, 'Name', sprintf('Image %d', ii));
  pause;
end
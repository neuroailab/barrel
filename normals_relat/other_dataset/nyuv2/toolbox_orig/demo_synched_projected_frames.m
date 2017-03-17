% The directory where you extracted the raw dataset.
% datasetDir = '[PATH TO THE NYU DEPTH V2 RAW DATASET]';
datasetDir = '/Users/chengxuz/barrel/bullet/barrle_related_files/nyuv2/';

% The name of the scene to demo.
% sceneName = '[NAME OF A SCENE YOU WANT TO VIEW]';
sceneName = 'study_0005';

% The absolute directory of the 
sceneDir = sprintf('%s/%s', datasetDir, sceneName);

% Reads the list of frames.
frameList = get_synched_frames(sceneDir);

% Displays each pair of synchronized RGB and Depth frames.
for ii = 1 : 15 : numel(frameList)
  imgRgb = imread([sceneDir '/' frameList(ii).rawRgbFilename]);
  imgDepth_bs = imread([sceneDir '/' frameList(ii).rawDepthFilename]);
  imgDepthRaw = swapbytes(imread([sceneDir '/' frameList(ii).rawDepthFilename]));
  
  disp(frameList(ii).rawDepthFilename)
  %disp(imgDepth_bs(1,1))
  
  figure(1);
  % Show the RGB image.
  subplot(1,3,1);
  imagesc(imgRgb);
  axis off;
  axis equal;
  title('RGB');
  
  % Show the Raw Depth image.
  subplot(1,3,2);
  imagesc(imgDepthRaw);
  axis off;
  axis equal;
  title('Raw Depth');
  caxis([800 1100]);
  
  % Show the projected depth image.
  imgDepthProj = project_depth_map(imgDepthRaw, imgRgb);
  subplot(1,3,3);
  imagesc(imgDepthProj);
  axis off;
  axis equal;
  title('Projected Depth');
  
  pause();
end

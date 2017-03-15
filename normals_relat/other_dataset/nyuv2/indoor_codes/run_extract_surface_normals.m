% Loads all of the in-painted depth images and estimates the surface
% normals for each pixel.
addpath('common/');
addpath('surfaces/');
addpath(genpath('iccv07Final'));
Consts;
Params;

% Whether or not to overwrite existing files.
OVERWRITE = false;

[projectionMask, projectionSize] = get_projection_mask();

%%
if ~exist(consts.surfaceNormalDataDir, 'dir')
  mkdir(consts.surfaceNormalDataDir);
end

for ii = 1 : consts.numImages
  fprintf('Extracting surface normals %d/%d.\n', ii, consts.numImages);

  if ~consts.useImages(ii)
    continue;
  end
  
  outFilename = sprintf(consts.surfaceNormalData, ii);
  if exist(outFilename, 'file') && ~OVERWRITE
    continue
  end
  
  load(sprintf(consts.imageDepthFilename, ii), 'imgDepthOrig');
  [H, W] = size(imgDepthOrig);
  
  % Use rgb_plane2rgb_world since the depth has already been projected onto
  % the RGB image plane.
  points3d = rgb_plane2rgb_world(imgDepthOrig);
  points3d = points3d(projectionMask,:);
  
  % Note that we need to swap X and Y here because compute_local_planes
  % keeps XZ to be the ground plane, just like in the image-plane.
  X = points3d(:,1);
  Y = points3d(:,3);
  Z = points3d(:,2);
  
  [imgPlanes, imgNormals, normalConf] = ...
      compute_local_planes(X, Z, Y, params);
  save(outFilename, 'imgPlanes', 'imgNormals', 'normalConf');
end

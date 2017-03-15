% Extracts (but does not aggregate) boundary-classifier features.
function extract_boundary_classifier_features_and_labels(stage, params, startNdx, endNdx)
  Consts;

  if nargin < 3
    startNdx = 1;
  end
  
  if nargin < 4
    endNdx = consts.numImages;
  end
  
  fprintf('\nExtracting boundary features for images %d-%d:\n', startNdx, endNdx);
  fprintf('(Be patient! Slow for the first few stages...)\n');
  for ii = startNdx : endNdx
    fprintf('Extracting boundary features %d/%d (Stage %d)\r', ...
        ii, consts.numImages, stage);
      
    if ~consts.useImages(ii)
      continue;
    end
      
    boundaryFeaturesFilename = sprintf(consts.boundaryFeaturesFilename, ...
      params.seg.featureSet, stage, ii);
    if exist(boundaryFeaturesFilename, 'file') && ~params.overwrite
      fprintf('Skipping image (it already exists) %d/%d\n', ii, consts.numImages);
      continue;
    end
    
    if stage == 1
      prevBoundaryDataFilename = sprintf(consts.watershedFilename, ii);
    elseif stage <= 3 && ...
        (params.seg.featureSet == consts.BFT_RGBD_SUP || ...
         params.seg.featureSet == consts.BFT_RGBD_SUP_SC)
      prevBoundaryDataFilename = sprintf(consts.boundaryInfoPostMerge, ...
        consts.BFT_RGBD, stage-1, ii);
    else
      prevBoundaryDataFilename = sprintf(consts.boundaryInfoPostMerge, ...
        params.seg.featureSet, stage-1, ii);
    end
    
    load(prevBoundaryDataFilename, 'boundaryInfo');
    
    load(sprintf(consts.imageRgbFilename, ii), 'imgRgb');
    load(sprintf(consts.planeDataFilename, ii), 'planeData'); 
    load(sprintf(consts.watershedFilename, ii), 'pbAll');
    load(sprintf(consts.objectLabelsFilename, ii), 'imgObjectLabels');
    load(sprintf(consts.instanceLabelsFilename, ii), 'imgInstanceLabels');

    [~, instanceLabels] = get_labels_from_instances(...
        boundaryInfo.imgRegions, imgObjectLabels, imgInstanceLabels);
    
    [boundaryFeatures, boundaryLabels] = get_boundary_classifier_features(...
        ii, imgRgb, planeData, boundaryInfo, pbAll, instanceLabels, params);
    save(boundaryFeaturesFilename, 'boundaryFeatures', 'boundaryLabels');
  end
  
  fprintf('\n');
  fprintf('===========================================\n');
  fprintf('Finished extacting boundaries for stage %d!\n', stage);
  fprintf('===========================================\n');
end



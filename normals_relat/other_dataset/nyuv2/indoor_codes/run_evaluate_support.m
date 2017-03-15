addpath('common/');
addpath('support/');
Consts;
Params;

params.regionSrc = consts.REGION_SRC_LABELS;
params.stage = 5;
params.seg.featureSet = consts.BFT_RGBD;

load(consts.splitsPath, 'trainNdxs');

load(consts.supportLabels);

%% 
% Initialize the confusion matrices for measuring structure-class 
% prediction and support label metrics.
confMatTrainPix = zeros(4);
confMatTestPix = zeros(4);

confMatTrainInst = zeros(4);
confMatTestInst = zeros(4);

evalRecords = zeros(consts.numImages, 4);

%%
for ii = 1 : consts.numImages
  if ~consts.useImages(ii)
    continue;
  end
  
  fprintf('Evaluating results %d/%d.\n', ii, consts.numImages);
  
  % Load the RGB image, the structure labels and the results file for the
  % current image.
  load(sprintf(consts.imageRgbFilename, ii), 'imgRgb');
  load(sprintf(consts.structureLabelsFilename, ii), 'imgStructureLabels');
  
  if ~exist(sprintf(consts.resultsIpFilename, params.regionSrc, ii), 'file')
    continue;
  end
  
  load(sprintf(consts.resultsIpFilename, params.regionSrc, ii), ...
      'supportLabelsPred', 'M', 'S');
  
  % Evaluate the support predictions.
  [numMatch correctSupportTypeAgnostic, correctSupportTypeAware,...
    missingFloorCorrect, observedFloorCorrect, ambiguousFloor] = ...
    evaluate_support_predictions(supportLabels{ii}, supportLabelsPred);
    
  % Get the ground truth and inferred structure labels.
  imgRegions = get_regions(ii, params);
  structureLabelsGt = get_labels_from_regions(imgRegions, imgStructureLabels);
  [~, structureLabelsLp] = max(M, [], 2);

  imgStructureLabelsPred = fill_regions_with_values(imgRegions, structureLabelsLp);
  [~, ~, ~, confMat] = eval_seg(imgStructureLabelsPred, imgStructureLabels, 4);
  
  % Accrue stats.
  if isin(ii, trainNdxs)
    confMatTrainPix = confMatTrainPix + confMat;
    confMatTrainInst = confMatTrainInst + confusion_matrix(structureLabelsGt, structureLabelsLp, 4);
  else
    confMatTestPix = confMatTestPix + confMat;
    confMatTestInst = confMatTestInst + confusion_matrix(structureLabelsGt, structureLabelsLp, 4);
  end
  
  evalRecord = [ ...
    isin(ii, trainNdxs), ...
    numMatch, ...
    nnz(correctSupportTypeAgnostic), ...
    nnz(correctSupportTypeAware), ...
  ];
  evalRecords(ii, :) = evalRecord;
end

%%

isTrain = false(consts.numImages, 1);
isTrain(trainNdxs) = true;

% Support Type Agnostic
fprintf('\n');
fprintf('  Type       Train    Test\n');
fprintf('Agnostic:   %2.1f    %2.1f\n', ...
  100 * sum(evalRecords(isTrain, 3)) ./ sum(evalRecords(isTrain, 2)), ...
  100 * sum(evalRecords(~isTrain, 3)) ./ sum(evalRecords(~isTrain, 2)));

% Support Type Aware
fprintf('Aware:      %2.1f  %2.1f\n', ...
  100 * sum(evalRecords(isTrain, 4)) ./ sum(evalRecords(isTrain, 2)), ...
  100 * sum(evalRecords(~isTrain, 4)) ./ sum(evalRecords(~isTrain, 2)));

fprintf('\n');


%%

% Calculate accuracy.
accTrain = sum(diag(confMatTrainPix)) / sum(confMatTrainPix(:));
accTest = sum(diag(confMatTestPix)) / sum(confMatTestPix(:));
fprintf('Acc Train (Pix)): %f\n', accTrain);
fprintf('Acc Test (Pix): %f\n', accTest);

% Calculate mean diagonal.
fprintf('Mean diag (Train-Pix): %f\n', mean(diag(normalize_conf_mat(confMatTrainPix))));
fprintf('Mean diag (Test-Pix): %f\n', mean(diag(normalize_conf_mat(confMatTestPix))));

% Calculate accuracy.
accTrain = sum(diag(confMatTrainInst)) / sum(confMatTrainInst(:));
accTest = sum(diag(confMatTestInst)) / sum(confMatTestInst(:));
fprintf('Acc Train (Inst)): %f\n', accTrain);
fprintf('Acc Test (Inst): %f\n', accTest);

% Calculate mean diagonal.
fprintf('Mean diag (Train-Inst): %f\n', mean(diag(normalize_conf_mat(confMatTrainInst))));
fprintf('Mean diag (Test-Inst): %f\n', mean(diag(normalize_conf_mat(confMatTestInst))));




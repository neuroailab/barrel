% Learns a dictionary of atoms used for sparse coding SIFT descriptors.
addpath('common/');
Consts;
Params;

addpath(consts.spamsPath);

%% Load the SIFT descriptor dataset.
fprintf('Loading SIFT dataset...');
datasetFilename = sprintf(consts.siftDataset, params.sift.patchSize, ...
    params.sift.stride, params.sift.normMethod);
load(datasetFilename, 'trainData');
fprintf('DONE.\n');
  
%%
fprintf('Learning dictionary of size %d...\n', params.sc.K);
scParams = struct();
scParams.D = rand(size(trainData, 2), params.sc.K);
scParams.mode = 2;

% Constrain the coefficients to be positive.
scParams.posAlpha = 1;
scParams.lambda = params.sc.lambda; % Coefficient for L1 Regularizer.
scParams.lambda2 = 0; % Coefficient for L2 Regularizer.
scParams.iter = 30;

% Max number of threads.
scParams.numThreads = 4;

D = mexTrainDL(trainData', scParams);
fprintf('DONE.\n');

%%
fprintf('Saving dictionary...');
save(sprintf(consts.siftDictionary, params.sift.patchSize, ...
    params.sift.stride, params.sift.normMethod, ...
    params.sc.K, params.sc.lambda), 'D');
fprintf('DONE.\n');


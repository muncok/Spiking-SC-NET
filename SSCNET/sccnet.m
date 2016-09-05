clear;
addpath(genpath('.'));
addpath(genpath('../'));
rand('state',10);
%% sc opts
scOpt.scRemoveZeroParam = true;
scOpt.scUsedScaledParam = true;
scOpt.scApproxDerandLevel = 0; % 0 means the exact count;
scOpt.scBitWidthMin = 1024;
scOpt.scBitWidthMax = 1024;
scOpt.scBitWidthOffset = 32;
%% load net & input
load('spike_stream.mat');
net = net_init('useBnorm', 0);
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'error');
net_ = net.saveobj();
obj = dagnn.DagNN.loadobjSC(net_, scOpt.scBitWidthMax, scOpt) ;
%% eval net
% load imdb.mat;


% testResult.trial = repmat(struct(...
%     'scBit', 0, ...
%     'scErrorNum', 0, ...
%     'scErrorRate', 0, ...
%     'binErrorNum', 0, ...
%     'binErrorRate', 0), ...
%     1, numel(num_sc_test));

% testResult.binerror = false(numel(num_sc_test), test_index_size);
% testResult.binpredic = cell(numel(num_sc _test), test_index_size);
% testResult.scerror = false(numel(num_sc_test), test_index_size);
% testResult.scpredic = cell(numel(num_sc_test), test_index_size);
% testResult.labels = images.labels(1,test_index_start:test_index_end);

batchSize = 100;
test_index_start = 1;
test_index_end   = 100;
test_index_size  = test_index_end - test_index_start + 1;
num_fail = 0;
num_success = 0;
num_sc_fail = 0;
num_sc_success = 0;
num_sc_test = scOpt.scBitWidthMin : scOpt.scBitWidthOffset : scOpt.scBitWidthMax;
inputs = struct('bin_input',bin_input,...
    'sc_input',sc_input,...
    'label', label');
%  inputs = struct('bin_input',bin_input,...
%                         'sc_input',sc_input,...
%                         'rate_input',rate_input,...
%                         'label', label');
for scBit = num_sc_test
    for i = test_index_start:batchSize:test_index_end
        batch = i:(i+batchSize-1);
        batch_bin = inputs.bin_input(:,:,:,batch);
        batch_sc = inputs.sc_input(:,:,:,:,batch);
        batch_label = inputs.label(1,batch) ;
        batch_inputs = struct('bin_input', batch_bin, 'sc_input', batch_sc, ...
            'label', batch_label) ;
        
        
        % evaluate SSC
        obj.evalSSC(batch_inputs,scOpt.scBitWidthMax);
        
        % statistics
        binerror = obj.vars(obj.getVarIndex('error')).value;
        scerror = obj.vars(obj.getVarIndex('error')).scvalue;
        
        num_fail = num_fail + binerror;
        num_success = num_success + (batchSize - binerror);
        
        num_sc_fail = num_sc_fail + scerror;
        num_sc_success = num_sc_success + (batchSize - scerror);
                fprintf('[Level-%d Appx][%d scbit] SC error %f (%d / %d) : Ref %f (%d / %d)\n', ...
            scOpt.scApproxDerandLevel, scOpt.scBitWidthMax, ...
            num_sc_fail / (num_sc_fail + num_sc_success), ...
            num_sc_fail, (num_sc_fail + num_sc_success), ...
            num_fail / (num_fail + num_success), ...
            num_fail, (num_fail + num_success));
        %--------------------------------------------------------------------------
        %         indexBatch = batch - test_index_start + 1;
        %
        %         scores = obj.vars(obj.getVarIndex('prediction')).value ;
        %         scores = squeeze(gather(scores)) ;
        %         value_scores = scores;
        %         testResult.binpredic(indexTrial, indexBatch) = mat2cell(scores, 10, ones(1,10));
        %         testResult.binpredic(indexTrial, indexBatch) = mat2cell(scores, 10, ones(1,batchSize));
        %
        %         show the classification results
        %         [bestScore, best] = max(scores) ;
        %
        %         label = obj.vars(obj.getVarIndex('label')).value ;
        %         success = (best == label);
        %
        %         testResult.binerror(indexTrial, indexBatch) = success;
        %         -----------
        %         scores = obj.vars(obj.getVarIndex('prediction')).scvalue ;
        %         scores = squeeze(gather(scores)) ;
        %         testResult.scpredic(indexTrial, indexBatch) = mat2cell(scores, 10, ones(1,10));
        %         testResult.scpredic(indexTrial, indexBatch) = mat2cell(scores, 10, ones(1,batchSize));
        %
        %         show the classification results
        %
        %
        %         [bestScore , best] = max(scores) ;
        %         label = obj.vars(obj.getVarIndex('label')).scvalue ;
        %         success = (best == label);
        %
        %         testResult.scerror(indexTrial, indexBatch) = success;
        %--------------------------------------------------------------------------
        
    end
end
%% error inspect
% args = struct();
% args.type = 'var';
% args.name = 'x2';
% error = obj.convertError(args);
% fprintf('average conversion error of %s : %f\n',args.name, error);


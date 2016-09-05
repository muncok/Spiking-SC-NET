addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
rand('state', 10);
%% dataset
load mnist_uint8;
nb_test = 1000;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);
rand_idx = randsample(size(test_x,1), nb_test);
mini_test_x = test_x(rand_idx,:);
mini_test_y = test_y(rand_idx,:);
%% initialize
% Initialize net
nn = nnsetup([784 200 200 10]);
% Rescale weights for ReLU
% for i = 2 : nn.n
%     % Weights - choose between [-0.1 0.1]
%     nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
%   a a  nn.vW{i - 1} = zeros(size(nn.W{i-1}));
% end
nn.W{1, 1} = h5read('400_weights.h5','/dense_13/dense_13_W');
nn.W{1, 2} = h5read('400_weights.h5','/dense_14/dense_14_W');
nn.W{1, 3} = h5read('400_weights.h5','/dense_15/dense_15_W');
% nn.W{1, 1} = h5read('pruned_weights.h5','/weights/W1')';
% nn.W{1, 2} = h5read('pruned_weights.h5','/weights/W2')';
% nn.W{1, 3} = h5read('pruned_weights.h5','/weights/W3')';
%% Data-normalize the NN
nn.activation_function = 'relu';
nn.output ='relu';
nn=nnff(nn, mini_test_x, mini_test_y);
% [norm_nn, norm_constants] = normalize_nn_data(nn, mini_test_x);
% for idx=1:numel(norm_constants)
%     fprintf('Normalization Factor for Layer %i: %3.5f\n',idx, norm_constants(idx));
% end
% fprintf('NN normalized.\n');
% norm_nn=nnff(norm_nn, mini_test_x, mini_test_y);
norm_nn = nn;
%% Test the Data-Normalized NN
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =   1.0;
t_opts.dt           = 0.001;
t_opts.duration     = 1.024;
t_opts.report_every = 0.001;
t_opts.max_rate     =  1000;
t_opts.record_layer =     2;
t_opts.nb_test      = nb_test;

norm_nn = spikeff(norm_nn, mini_test_x, t_opts);
%% post-processing for SC
nb_timesteps = t_opts.duration / t_opts.dt;
rate_code =norm_nn.layers{1,2}.sum_spikes / nb_timesteps;
% rate_code = rate_code/ norm(rate_code);
% float_num = nn.a{1,2}(1,:)';
% float_num = float_num / max(float_num);
% float_num = float_num/ norm(float_num);
% compare = [rate_code(1,:); float_num'];
% fprintf('rate_code : %f,  float_num : %f',rate_code,float_num);
% scale_f = norm(float_num) / norm(rate_code);
% fprintf('scale_f : %f\n', scale_f);
% % rate_code = rate_code .* scale_f;
% bitstream = num2str(nn.outspikes, '%1d');
% bitstream = strcat(bitstream, '| ',num2str(rate_code'), '| ', num2str(float_num));
% bitstream = strcat(num2str(rate_code'), '| ', num2str(float_num));

rec_spikes = reshape(norm_nn.rec_spikes,1,[]);
% a = rand(size(rec_spikes)) >= 0.5;
% rec_spikes_ = or(rec_spikes, a);
% rec_spikes_ = reshape(rec_spikes_, size(norm_nn.rec_spikes));
% after = (sum(rec_spikes_(:,:,1), 1) - 512) / 512;
% after_bin = rate_code(1,:);
% test_after = [after; after_bin];

%sc input
sc_input = uint8(reshape(rec_spikes,nb_timesteps,[]));
sc_input = bwpack(sc_input);
sc_input = reshape(sc_input, 32, 1, 1,200,nb_test);
% 
bin_act = nn.a{1,2}';
bin_input = single(reshape((2*bin_act - 1), 1,1,200,nb_test));
rate_input = single(reshape((2*rate_code' - 1), 1,1,200,nb_test));
[~,label] = max(mini_test_y,[], 2);

% for i = [2 3]
%     weight_std = std(reshape(nn.W{1,i}, 1, []));
%     dropThreshold = weight_std * 0.1;
%     nearZero(i) = sum(sum(abs(nn.W{1,i}) > dropThreshold))/numel(nn.W{1,i});
% end
%% save weights and spikes
weights = struct('W_1', nn.W{1,1},...
    'W_2', nn.W{1,2},...
    'W_3', nn.W{1,3});
save('rec_spike.mat', 'bin_input', 'sc_input', 'rate_input',...
    'label',...
    'weights');

% save('rate_spike_10000.mat', 'rate_code', 'train_y');
% h5create('mnist_ratecode.h5', '/rate_code', size(rate_code));
% h5create('mnist_ratecode.h5', '/label', size(mini_test_y));
% h5write('mnist_ratecode.h5' , '/rate_code', 2*rate_code - 1);
% h5write('mnist_ratecode.h5', '/label', mini_test_y);
fprintf('Done.\n');

%% histogram for activation
% figure(2);
% hist(reshape(rate_input,1,[]));
% h = findobj(gca,'Type','patch');
% set(h,'FaceColor','r','EdgeColor','w','facealpha',0.75)
% hold on;
% hist(reshape(bin_input/max(max(bin_act)),1,[]));
% h1 = findobj(gca,'Type','patch');
% set(h1, 'facealpha',0.75);

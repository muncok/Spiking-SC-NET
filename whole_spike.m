%% environment setup
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
rand('state', 10);
%% load data
load mnist_uint8;
nb_test = 10000;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);
%% NN setup
nn = nnsetup([784 200 200 10]);
nn.W{1, 1} = h5read('nonzero_weights.h5','/dense_1/dense_1_W');
nn.W{1, 2} = h5read('nonzero_weights.h5','/dense_2/dense_2_W');
nn.W{1, 3} = h5read('nonzero_weights.h5','/dense_3/dense_3_W');
nn.activation_function = 'relu';
nn.output ='relu';
%% NN feed-forward
nn = nnff(nn, test_x, test_y);
%% spikingNN feed_forward
record_layer = 2;
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =   1.0;
t_opts.dt           = 0.001;
t_opts.duration     = 1.024;
t_opts.report_every = 0.001;
t_opts.max_rate     =  1000;
t_opts.record_layer = record_layer;
t_opts.nb_test      = nb_test;

nn = spikeff(nn, test_x, t_opts);
%% post-processing
nb_timesteps = t_opts.duration / t_opts.dt;

% rate_code
% rate_code =nn.layers{1, record_layer}.sum_spikes / nb_timesteps;
% rec_spikes = reshape(nn.rec_spikes,1,[]);

% sc_input
record_dim = size(rec_code,2);
sc_input = uint8(reshape(rec_spikes,nb_timesteps,[]));
sc_input = bwpack(sc_input);
bwpack_row = nb_timesteps / 32;
sc_input = reshape(sc_input, bwpack_row , 1, 1,record_dim, nb_test);

bin_act = nn.a{1, record_layer}';
bin_input = single(reshape((2*bin_act - 1), 1,1,record_dim, nb_test));
rate_input = single(reshape((2*rate_code' - 1), 1,1,record_dim, nb_test));
[~,label] = max(test_y,[], 2);
%% save data
weights = struct('W_1', nn.W{1,1},...
                 'W_2', nn.W{1,2},...
                 'W_3', nn.W{1,3});
             
save('spike_stream.mat', ...
     'bin_input', 'sc_input',...
     'label',...
     'weights');
      
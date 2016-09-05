function net = net_init(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST

opts.useBnorm = true ;
% KKHTEST
% opts.useBnorm = false;

opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

scaletanh = 1;

%f=1/100 ;
f=1/5 ;
%f=1; %%%% fail!
%f=2/3 ;
b = 0;

net.layers = {} ;
% layer 1,2
% f*randn(28,28,1,200, 'single')
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{reshape(weights.W1',, b*ones(1,200,'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0, ...
%                            'scrangeweight', 0, ...
%                            'verbose', 0) ;
% %'weights', {{f*(2*rand(28,28,1,100, 'single')-1), []}}, ...                        
% %'weights', {{f*randn(28,28,1,100, 'single'), []}}, ...
% net.layers{end+1} = struct('type', 'tanh') ;
% 
% layer 3,4
% W_3_ = reshape(weights.W_2',1,1,200,200);
W_3 = h5read('400_weights.h5','/dense_14/dense_14_W');
W_3_ = single(reshape(W_3',1,1,200,200));
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{W_3_, b*ones(1,200,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'scrangeweight', 0, ...
                            'verbose', 0) ;
%'weights', {{f*(2*rand(28,28,1,100, 'single')-1), []}}, ...                        
%'weights', {{f*randn(28,28,1,100, 'single'), []}}, ...
net.layers{end+1} = struct('type', 'tanh') ;

% layer 5,6
% W_5_ = reshape(weights.W_3',1,1,200,10);
W_5 = h5read('400_weights.h5','/dense_15/dense_15_W');
W_5_ = single(reshape(W_5',1,1,200,10));
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{W_5_, b*ones(1,10,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'scrangeweight', 0, ...
                            'verbose', 0) ;
%'weights', {{f*(2*rand(1,1,200,10, 'single')-1), []}}, ...
%'weights', {{f*randn(1,1,200,10, 'single'), []}}, ...
net.layers{end+1} = struct('type', 'tanh', 'scale', scaletanh) ;

% dummy-layers for SC
net.layers{end+1} = struct('type', 'sc2bin') ;                        
net.layers{end+1} = struct('type', 'softmaxloss') ;

% optionally switch to batch normalization
if opts.useBnorm
  net = insertBnorm(net, 1) ;
  net = insertBnorm(net, 4) ;
  net = insertBnorm(net, 7) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;

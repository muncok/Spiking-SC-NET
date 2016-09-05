function nn=spikeff(nn, test_x, opts)
dt = opts.dt;
num_examples = size(test_x,1);

% Initialize network architecture
for l = 1 : numel(nn.size)
    blank_neurons = zeros(num_examples, nn.size(l));
    nn.layers{l}.mem = blank_neurons;
    nn.layers{l}.refrac_end = blank_neurons;        
    nn.layers{l}.sum_spikes = blank_neurons;
end

% nn.bitstream = cell(size(test_x,1), 1);
% nn.bitstream(:) = {zeros(,numel(dt:dt:opts.duration))};
rec_layer = opts.record_layer;
nn.rec_spikes = zeros(opts.duration/opts.dt, size(nn.W{1,opts.record_layer}, 1), opts.nb_test);

% Time-stepped simulation
iter = 1;
for t=dt:dt:opts.duration
        % Create poisson distributed spikes from the input images
        %   (for all images in parallel)
        rescale_fac = 1/(dt*opts.max_rate);
        spike_snapshot = rand(size(test_x)) * rescale_fac;
        inp_image = spike_snapshot <= test_x;

        nn.layers{1}.spikes = inp_image;
        nn.layers{1}.sum_spikes = nn.layers{1}.sum_spikes + inp_image;
        for l = 2 : numel(nn.size)
            % Get input impulse from incoming spikes--
            impulse = nn.layers{l-1}.spikes*nn.W{l-1}';
            % Add input to membrane potential
            nn.layers{l}.mem = nn.layers{l}.mem + impulse;
            % Check for spiking 
            nn.layers{l}.spikes = nn.layers{l}.mem >= opts.threshold;
            % Reset
            nn.layers{l}.mem(nn.layers{l}.spikes) = 0;
            % Ban updates until....
            nn.layers{l}.refrac_end(nn.layers{l}.spikes) = t + opts.t_ref;
            % Store result for analysis later
            nn.layers{l}.sum_spikes = nn.layers{l}.sum_spikes + nn.layers{l}.spikes;            
        end
        outspikes = nn.layers{rec_layer}.spikes';
        nn.rec_spikes(iter,:,:) = outspikes;
        
%         if(mod(round(t/dt),round(opts.report_every/dt)) == round(opts.report_every/dt)-1)
%             
%         else
%             fprintf('.');            
%         end
        iter = iter + 1;
end
    
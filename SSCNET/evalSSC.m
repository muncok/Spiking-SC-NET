function evalSSC(obj, inputs, scBitWidth)

obj.computingDerivative = false;
obj.conserveMemory = false;
% -------------------------------------------------------------------------
% Forward pass
% -------------------------------------------------------------------------

% set the input values

%scBitWidth = 1024;
input_idx = obj.getVarIndex('input');
label_idx = obj.getVarIndex('label');

bin_input = inputs.bin_input;
bin_scale = max(max(bin_input));
% rate_input = inputs.rate_input;

% value
obj.vars(input_idx).value = bin_input / bin_scale; % input
% obj.vars(input_idx).value = rate_input;

% scvalue
% obj.vars(input_idx).scvalue = dagnn.bin2ScIdeal(bin_input / bin_scale, scBitWidth) ; % input
% obj.vars(input_idx).scvalue = dagnn.bin2ScIdeal(rate_input, scBitWidth);
obj.vars(input_idx).scvalue = inputs.sc_input;

% label
obj.vars(label_idx).value = inputs.label ;
obj.vars(label_idx).scvalue = inputs.label ; 


% figure(872);
% subplot(3,1,1);
% one = reshape(inputs{2}, 1, []);
% bar(one);
% subplot(3,1,2);
% univalue = sum(obj.vars(v(1)).scvalue) / scBitWidth;
% bivalue = 2 * univalue - 1;
% two = reshape(bivalue, 1, []);
% bar(two);
% subplot(3,1,3);
% bar(one - two);


obj.numPendingVarRefs = [obj.vars.fanout] ;
for l = obj.executionOrder
  time = tic ;
  obj.layers(l).block.forwardAdvancedSC(obj.layers(l)) ;
  obj.layers(l).forwardTime = toc(time) ;
end

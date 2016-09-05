function [error] = convertError(obj, opts)
%SCCNET_UTILS Summary of this function goes here
%   Detailed explanation goes here
if strcmp(opts.type,'var')
    var = opts.name;
    Value = obj.vars(obj.getVarIndex(var)).value;
    sc_number = obj.vars(obj.getVarIndex(var)).scvalue;
else
    param = opts.name;
    Value = obj.params(obj.getParamIndex(param)).value;
    sc_number = obj.params(obj.getParamIndex(param)).scvalue;
end
    
scValue = dagnn.sc2BinIdeal(sc_number);
error = sum(sum(abs(Value- scValue))) / numel(nonzeros(Value));     
end


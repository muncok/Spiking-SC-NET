function scNum = bin2sc (binNum, scBits)
    scNum = zeros(size(binNum,1), scBits);
    for i = 1:size(binNum,1)
        idx = randperm(scBits, binNum(i));
        scNum(i,idx) = 1;
    end
end
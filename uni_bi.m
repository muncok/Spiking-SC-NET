scBits = 1024;
x = 0:0.01:1;
uni = zeros(length(x),1);
bi = zeros(length(x),1);
for i = 1:length(x);
    x_ = x(i);
    uni(i) = round(x_ * scBits);
    bi(i) = round((x_ + 1)./ 2 * scBits);
end
uni_sc = bin2sc(uni, scBits);
bi_sc = bin2sc(bi, scBits);

uni_sc_  = reshape(uni_sc, 1,[]);
halfsc = rand(size(uni_sc_)) <= 0.5;
uni_sc_ = or(uni_sc_, halfsc);
uni_sc_ = reshape(uni_sc_,size(uni_sc));

error = (sum(bi_sc,2) - sum(uni_sc_,2))/scBits;
error2 = (sum(bi_sc,2) - sum(uni_sc,2))/scBits;
figure(2);
stem(x, error);
title('error w/ OR gate');

figure(3);
stem(x, error2);
title('error w/o OR gate');


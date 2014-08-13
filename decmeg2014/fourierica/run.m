PAGER('/dev/null');
page_screen_output(1);
page_output_immediately(1);

source fourierica.m
load /home/rodion/decmeg2014/traindata/train_subject01.mat
options.samplfreq=250;
options.pcadim=40;
options.minfreq=7;
options.maxfreq=30;
options.windowlength_sec=0.1;
options.complexmixing=false;


result = zeros(size(X)(1), 40, 375);
for i = 1:size(X)(1)
    XX = reshape(X(i,:,:), 306, 375);
    [S_FT,A_orig,W_orig] = fourierica(XX, options);
    XX_new = W_orig * XX;
    result(i, :, :) = XX_new;
endfor

save /home/rodion/decmeg2014/traindata/fourierica/1.mat result;
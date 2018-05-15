%%
clear;clc;close all;
load chirp
t = (0:length(y)-1)/Fs;
%% high requency filter
bhi = fir1(34,0.48,'high');
freqz(bhi,1)
title('Filter in frequency domain')
xlabel('');
subplot(2,1,2)
stem(bhi);
title('Filter in time domain');
ylabel('Magnitude');
%%
figure;
outhi = filter(bhi,1,y);
subplot(2,1,1)
plot(t,y)
title('Original Signal')
ys = ylim;

subplot(2,1,2)
plot(t,outhi)
title('Highpass Filtered Signal')
xlabel('Time (s)')
ylim(ys)

%%
blo = fir1(34,0.48,chebwin(35,30));
freqz(blo,1)
title('Filter in frequency domain')
xlabel('');
subplot(2,1,2)
stem(blo);
title('Filter in time domain');
ylabel('Magnitude');
%%
outlo = filter(blo,1,y);

subplot(2,1,1)
plot(t,y)
title('Original Signal')
ys = ylim;

subplot(2,1,2)
plot(t,outlo)
title('Lowpass Filtered Signal')
xlabel('Time (s)')
ylim(ys)

%%
ord = 44;
low = 0.4;
bnd = [0.6 0.9];
bM = fir1(ord,[low bnd]);

freqz(bM,1)
title('Filter in frequency domain')
xlabel('');
subplot(2,1,2)
stem(bM);
title('Filter in time domain');
ylabel('Magnitude');
%%

outlo = filter(bM,1,y);

subplot(2,1,1)
plot(t,y)
title('Original Signal')
ys = ylim;


subplot(2,1,2)
plot(t,outlo)
title('bandPass Filtered Signal')
xlabel('Time (s)')
ylim(ys)

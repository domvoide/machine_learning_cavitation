clear;
clc;

data = load('Test_dyna_40kHz');
data_acc_tri = data.UntitledTri100gx_Y_SN2139607;
data_acc_uni = data.UntitledUni100g_Y_SN4880946;
data_hydro = data.UntitledHydrophone_SN5493417;

t = data.UntitledHydrophone_SN5493417_Time;

%% Signaux bruts

acc_tri = data_acc_tri - mean(data_acc_tri);
acc_uni = data_acc_uni - mean(data_acc_uni);
hydro = data_hydro - mean(data_hydro);

signaux_bruts = figure('Name','Micro pour alpha 14?',...
           'PaperSize',[20 20],'PaperPosition',[0 0 20 20],'Position',[500 200 500 400]);
       
subplot(3,1,1)
Acc_tri = plot(t,acc_tri);
title('Signal brut de l acc?l?rom?tre tri-axes');
xlabel('Time [sec]');
ylabel('Amplitude [g]');

subplot(3,1,2)
Acc_uni = plot(t,acc_uni);

subplot(3,1,3)
Hydro = plot(t,hydro);


%% FFT
L = length(acc_tri(2400001:2800000));
NFFT = 2^nextpow2(L);
Fs = 40000;
freq = Fs/2*linspace(0,1,NFFT/2+1);

% window = 20;
% b = (1/window)*ones(1,window);
% fdelay = length(1:b-1/2);

acc_tri_fft = fft(acc_tri(2400001:2800000),NFFT)/L;
acc_tri_amp(1:NFFT/2+1) = 2*abs(acc_tri_fft(1:NFFT/2+1));
% acc_tri_filt = filter(b,1,acc_tri_amp);

acc_uni_fft = fft(acc_uni(2400001:2800000),NFFT)/L;
acc_uni_amp(1:NFFT/2+1) = 2*abs(acc_uni_fft(1:NFFT/2+1));
% acc_uni_filt = filter(b,1,acc_uni_amp);

hydro_fft = fft(hydro(2400001:2800000),NFFT)/L;
hydro_amp(1:NFFT/2+1) = 2*abs(hydro_fft(1:NFFT/2+1));
% hydro_filt = filter(b,1,hydro_amp);


FFT = figure('Name','Micro pour alpha 14?',...
           'PaperSize',[10 20],'PaperPosition',[0 0 10 20],'Position',[500 500 500 200]);
       
subplot(3,1,1)
FFT_acc_tri = plot(freq,acc_tri_amp);
set(gca,'Ylim',[0,0.05]);
title('FFT de l acc?l?rom?tre tri axes');

subplot(3,1,2)
FFT_acc_uni = plot(freq,acc_uni_amp);
set(gca,'Ylim',[0,0.0085]);
title('FFT pour l acc?l?rom?tre uni axe')

subplot(3,1,3)
FFT_hydro = plot(freq,hydro_amp);
set(gca,'Ylim',[0,2400],'Xlim',[0,1000]);
title('FFT pour l hydrophone');

%% Spectrogrammes

spectro = figure('Name','Micro pour alpha 14?',...
           'PaperSize',[20 20],'PaperPosition',[0 0 20 20],'Position',[500 500 500 400]);
       
subplot(3,1,1)
spectrogram(acc_tri,5000,2500,50000,40000,'yaxis','power');
caxis([-30;-15]);
% view(-30,45)
% set(gca,'Zlim',[-80;0]);
colormap jet
title('Spectrogramme pour l acc?l?rom?tre tri-axes');


subplot(3,1,2)
spectrogram(acc_uni,5000,2500,50000,40000,'yaxis','power');
% view(-45,65)
colormap jet
caxis([-55;-40]);

subplot(3,1,3)
spectrogram(hydro,5000,2500,100000,40000,'yaxis','power');
% view(-45,65)
colormap jet
set(gca,'Ylim',[0;0.5]);
caxis([65;75]);


       
       
       
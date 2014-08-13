% script demo TFBSS
close all
clc
disp(' ')
disp('*************************************************************************')
disp('*                                                                       *')
disp('*                              TFBSS DEMO                               *')
disp('*                                                                       *')
disp('*************************************************************************')
disp('C.FEVOTTE (cedric.fevotte@irccyn.ec-nantes.fr) - Sep, 02 2002')
disp(' ')
disp('TFBSS performs blind source separation of linear instantaneous mixtures  ')
disp('of non-stationary sources.')
disp(' ')
disp('This demo performs separation of large-band overlapping Time-Varying ARMA')
disp('signals.')
disp('There are 3 sources and 4 observations of length T=256.')
disp('SNR is 20 dB (defined as the worst SNR of each source).')
disp(' ')

load data; % Load the sources
% Rows of S represent sources

[n,T]=size(S); % [Number of sources, Length of signals]
SNR=20; % SNR is defined as the worst SNR of each source

m=4; % Number of observations

% Parameters for computation of the Time-Frequency Distributions
Nt=T/2;
Nf=T/2;

disp('******************** The sources and their TFDs *************************') 
disp(' ')
disp('The Time-Frequency Distributions (TFDs) of the original sources are being')
disp('computed...')
disp('TFD is a Smoothed Pseudo Wigner-Ville Distribution.')
disp(' ')

tfr_s=cell(1,n);

for i=1:n
tfr_s{i}=tfrspwv(hilbert(S(i,:).'),1:T/Nt:T,Nf);
end

disp('Press any key to plot the sources with their TFDs.')
disp(' ')
pause;

figure(1);
for i=1:n
subplot(2,3,i);
plot(S(i,:)); xlabel('Time'); title(['Source ' int2str(i)]); axis tight
subplot(2,3,i+3);
imagesc([0:T],[0:1/T:0.5-1/T],tfr_s{i}); axis xy; xlabel('Time');
ylabel('Normalized Frequency');
title(['TFD of source ' int2str(i)]);
colorbar;
end

disp('****************************** Mixing ***********************************')
disp('Press any key to perform mixing.')
disp(' ')
pause;
disp('The mixing matrix A is chosen randomly:')
disp('A=')
A=randn(m,n);
disp(A)

disp(' ')
disp('Noise is added to the observations with SNR=')
disp(SNR)
disp('(defined as the worst SNR of each observation)')
disp(' ')

sigb=min(sqrt(sum(A.^2,2)))*exp(-SNR*log(10)/20); % Noise std
 
X=A*S+sigb*randn(m,T);

disp('Press any key to plot the observations.')
disp(' ')
pause;

figure(2);
for i=1:m
    subplot(m,1,i);
    plot(1:T,X(i,:));
    title(['Observation ' int2str(i)]); axis tight
end
xlabel('Time');

disp('**************************** Separation *********************************') 
disp(' ')
disp('Press any key to perform separation.')
disp(' ');
pause;
disp('Now TFBSS is performing separation... This is going to take a while.')
disp('The computation time depends on the chosen number of time instants Nt and')
disp('number of frequency bins Nf for computation of the TFDs.')
disp(' ')
disp('In this demo: Nt=Nf=T/2=128.')
disp(' ')
disp('For optimal results choose maximum resolution: Nt=Nf=T.')
disp(' ')
disp('ooooooooooooooooo TFBSS is working oooooooooooooooooo')

[Se,Ae]=tfbss(X,n,Nt,Nf);

disp('ooooooooooooooooooooooooooooooooooooooooooooooooooooo')
disp(' ')
disp(' ')
disp('***************************** Results ***********************************')
disp('Unmixing matrix: pinv(Ae)=')
disp(pinv(Ae))
disp(' ')
disp('Global system: pinv(Ae)*A=')
disp(pinv(Ae)*A)
disp('(Ideally this is a matrix with only one non-zero entry per row and column)')
disp(' ')

% Normalizing the original sources.
% TFBSS is shaped to provide zero mean unit variance sources: there is no need to
% normalize Se.
S_n=(diag(std(S.').^-1)*S);
% Computing the cross-correlation between real and estimated sources:
% This allows us to match the estimated sources with the original ones
C=1/T*(S_n*Se.');
Se_t=[];
for i=1:n
    [maxj,jmax]=max(abs(C(i,:)));
    Se_t=[Se_t; sign(C(i,jmax))*Se(jmax,:)];
end

disp('Estimated sources are compared to the original sources, after fixing     ')
disp('permutations, sign and scale.                                            ')
disp(' ')
disp('Press any key to plot the figure.')
disp(' ')
pause;


figure(3); 

for i=1:n
    subplot(n,1,i);
    plot(1:T, S_n(i,:).','b',1:T,Se_t(i,:).' ,'r');
    title(['Source ' int2str(i) ' (original: blue - estimated: red)']);
    axis tight
end

disp('***************************** End of demo *******************************')

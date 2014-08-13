function [Se,Ae]=tfbss(X,n,Nt,Nf,tol)
% TFBSS Blind Source Separation of (over)determined multiplicative mixtures
% 	of non-stationary real-valued sources. 
%
%   Usage: [Se,Ae]=tfbss(X,n,Nt,Nf,tol) 
%
%   Compulsory inputs:
%   - X: m x T matrix containing the m observation signals of length T
%   (rows represent signals);
%   - n: number of sources to be estimated.
%
%   Optional inputs:
%   - Nf: number of frequency bins used in the TFDs computation;
%   - Nt: number of equally spaced time instants in the TFDs 
%   computation;
%   - tol: gradient norm threshold.
%
%   -> Defaults parameters:
%   - if T < 256, Nt=T, Nf=closest power of 2 from T, else Nt=Nf=256; 
%   - tol=2/(Nt+Nf);
%
%   WARNINGS:
%   o For faster computation, it is highly recommended that Nf is a
%   power of two.
%
%   o Sources are assumed to be zero-mean, remove the mean of the
%   observations if necessary.
%
%   Outputs:
%   - Se: n x T matrix containing the n estimated sources,
%   - Ae: m x n estimated mixing matrix,
%
%   TFBSS is based on the joint-diagonalization of whitened and 
%   noise-compensated Spatial Time-Frequency Distributions (STFDs) matrices  
%   of the observations, corresponding to single auto-terms positions.
%
%   Main reference (available at http://www.irccyn.ec-nantes.fr/~fevotte):
%   A.HOLOBAR, C.FEVOTTE, C.DONCARLI, D.ZAZULA, "Single autoterms selection
%   for blind source separation in time-frequency plane", XIe EUSIPCO,
%   Toulouse, France
%
%   The iterative selection of maxima of the criterion in the above paper has
%   been replaced by a more simple gradient approach to be published 
%   soon (paper available upon request).
%   The inner Iterative Joint Diagonalization procedure is not  
%   implemented in this code.
%
%   Linked m-files: 
%   - joint_diag_rc.m, J.F Cardoso's slightly modified routine
%   joint_diag_r.m available at:
%   http://www.tsi.enst.fr/~cardoso/stuff.html
%   - tfrspwv.m, window.m, from the Matlab Time-Frequency Toolbox:
%   http://crttsn.univ-nantes.fr/~auger/tftb.html
%
%   Copyright (C) 01 Sep. 2003  C.FEVOTTE, A.HOLOBAR
%   Inquiries, bug report: fevotte@irccyn.ec-nantes.fr

%    This program is free software; you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation; either version 2 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program; if not, write to the Free Software
%    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

[m,T]=size(X); 

if n>m, fprintf('-> Number of sources must be lower than number of sensors\n'), return,end

if nargin == 2
  
  % Default values of Nt, Nf, tol
  if T<256,
    Nt=T; 
    Nf=2^floor(log2(T)); %Closest power of 2
  else Nf=256; Nt=256;
  end
  tol=2/(Nt+Nf);
  
elseif nargin == 4
  % Default value of tol 
  tol=2/(Nt+Nf);
end

if 2^nextpow2(Nf)~=Nf, 
fprintf('WARNING: For a faster computation, Nf should be a power of two\n');
end

verbose = 1; % Set to 0 for quiet operation

%============================ Whitening ==================================%     
if verbose, fprintf('-> Whitening the data\n'); end
Rxx=(1/T)*X*X.'; % Estimated time average power 
% (should be the same that Rxx=cov(X.',1));

[V,D]=eig(Rxx);
% Sorting the eigenvalues in ascending order 
[EigVal_Rxx,index]=sort(diag(D));
EigVect_Rxx=V(:,index);

  % Estimation of the variance of the noise
  if m>n
    noisevar=mean(EigVal_Rxx(1:m-n)); % Estimation possible if m>n
  else
    % If m=n, no estimation possible, assume noiseless environment 
    noisevar=0; 
  end
  
  % Whitening matrix
  fact=sqrt(EigVal_Rxx(m-n+1:m)-noisevar);
  for i=1:n      
      W(:,i)=(1/fact(i))*EigVect_Rxx(:,m-n+i);
  end
  W=W';
  
Z=W*X; % Whitened observation signals

%=================================================================================%  
  
  
%======================== Computation of the STFD of Z ===========================%
if verbose, fprintf('-> Computing STFD of the whitened observation signals\n'); end

Zh=hilbert(Z.').'; % Computation of the hilbert transform of Z
% This cancels negative frequencies of signals spectrum and then prevent from
% spectral folding of the TFDs (i.e produces an analytic signal).

STFDZ=zeros(n,n,Nf,Nt);

for i=1:n
    STFDZ(i,i,:,:)=tfrspwv(Zh(i,:).',ceil(1:T/Nt:T),Nf); % Auto-terms
end

for i=1:n
      for j=i+1:n          
          TFR=tfrspwv([ Zh(i,:).' Zh(j,:).'],ceil(1:T/Nt:T),Nf); % Cross-TFDs
          STFDZ(i,j,:,:)=TFR;
          STFDZ(j,i,:,:)=conj(TFR);
      end
end     
%==================================================================================%


%========================= Computation of the criterion C ========================%
if verbose, fprintf('-> Computing the criterion\n'); end
Tr=zeros(Nf,Nt);
C=zeros(Nf,Nt);

for f=1:Nf
	for t=1:Nt
       
	  STFDZ(:,:,f,t)=STFDZ(:,:,f,t)- noisevar*W*W'; % Noise compensation
	  Tr(f,t)=abs(trace(STFDZ(:,:,f,t))); % Forming trace

    end
end

meanTr=mean(mean(Tr)); % Mean value of Tr all over t-f plane

% The criterion is computed for points such that Tr>meanTr
Trthr=Tr>meanTr; % Thresholded trace t-f matrix
[F_trace,T_trace]=find(Trthr);

for k=1:length(F_trace)

      temp=abs(eig(STFDZ(:,:,F_trace(k),T_trace(k)))); % Computing eigenvalues

      if sum(temp)~=0
    	C(F_trace(k),T_trace(k))=max(temp)/sum(temp);
      else C(F_trace(k),T_trace(k))=0;
      end
      
end

%===================================================================================%

%================= Collecting single autoterms (t,f) positions =====================%
if verbose, fprintf('-> Collecting single auto-terms positions\n'); end
Jacneg=zeros(Nf,Nt);
Gsmall=zeros(Nf,Nt);
Maxpoints=zeros(Nf,Nt);

[Gt Gf]=gradient(C);
[Jtt Jtf]=gradient(Gt);
[Jft Jff]=gradient(Gf);

% Points where the 2-norm of gradient is small
Gsmall=sqrt(Gt.^2+Gf.^2)<tol;

% Points where the Jacobi is negative definite
D=Jtt.*Jff-Jtf.*Jft;
Jacneg=(D>0).*(Jtt<0).*((Jtt+Jff)<0);

Maxpoints=Gsmall.*Jacneg;

[F_grad,T_grad]=find(Maxpoints);
nbPoints=length(F_grad); % Number of points found
if nbPoints==0
    fprintf('-> No t-f location could be selected,\n')
    fprintf('   Please increase the gradient norm threshold\n')
    return
end
%===================================================================================%

%=================== Joint-Diagonalization of selected points ======================%
if verbose, fprintf('-> Joint-Diagonalization\n'); end
Rjd=[];

for i=1:nbPoints
   Rjd=[Rjd STFDZ(:,:,F_grad(i),T_grad(i))]; % Matrices to be joint-diagonalized
end

[U,D]=joint_diag_rc(Rjd,1e-8);
%===================================================================================%

%================ Forming estimated sources and mixing matrix ======================%
if verbose, fprintf('-> Over\n'); end
Se=U'*Z; % Rotation of the whitened observations
Ae=pinv(W)*U; % Estimation of the mixing matrix
%===================================================================================%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    DIARY   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 01 Sep 2003: slight notation modifications, added a warning, new default
% paramaters for Nt, Nf and tol (should be smarter).

% 22 Jan 2003: minor changes in help. Now signals are assumed to be
% zero-mean. Reduced Interference Distribution has been replaced by more
% computation friendly Smoothed Pseudo Wigner-Ville Distribution
% because it appeared through simulations that kernel has little
% influence, as long as the Wigner-Ville distribution is smoothed
% by some way. Joint_diag.m is replaced by joint_diag_rc.m
% (joint-diagonalization routine for complex matrices which have a
% REAL-VALUED common orthonormal basis).

% 9 Sept 2002: program stops if no t-f location is selected.

% 10 July 2002: minor changes in the presentation of the code, spell check.

% 27 May 2002: released in the public domain.

% 24 May 2002: optimization of the computation of the criterion
% C is now only computed for (t,f) locations such that Tr>mean(Tr).



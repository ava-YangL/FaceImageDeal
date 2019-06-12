function [ Z ] = Integration_FC( N, bkg, Slant, method, delta, phi )
%FRANKOTCHELLAPA Summary of this function goes here
%   Detailed explanation goes here

PQ = convPQ(N);
if Slant == 0
   Slant = -1;
end

if method == 'F'
   [Z] = ZFrankChelFFT(PQ,Slant,delta,phi);
else
   [Z] = ZFrankChelDCT(PQ,Slant,delta,phi);    
end

Z=Z.*bkg;

end

function PQ = convPQ(N)

[rows, cols, trash] = size(N);
PQ = zeros(rows,cols,2);
for x=1:rows
    for y=1:cols
        if N(x,y,3)==0
            PQ(x,y,1) = 0;
            PQ(x,y,2) = 0;
        else
            PQ(x,y,1) = (N(x,y,1)/N(x,y,3));
            PQ(x,y,2) = (N(x,y,2)/N(x,y,3));
        end    
    end
end

end

function [Z,DC] = ZFrankChelDCT(PQ,max_pq,Delta,Miu)

%
%
%   Global gradient field integration method proposed by 
%   Frankot and Chellappa in [1], but using the Discrete Cosine Transform
%   rather than the Fast Fourier Transform.
%  
%   Z = ZFrankChel(PQ,max_pq,med_,Delta,Miu) returns the height map Z
%   corresponding to the gradient field PQ, where PQ(:,:,1) =  gradient
%   in X direction and PQ(:,:,2) = gradient in Y direction.
%   max_pq filters the gradients that will be taken into account, it is 
%   almost always set as 4, but can be ignored depending on the set of
%   surface normals. Delta and Miu are regularizing terms 
%   referring to curvature and area constraints as proposed in [2].
%   when these parameters are both set to zero, the result is the same
%   as originally proposed in [1]
%   
%   A more complete explanation about implementing this method can be found in [3]
%
%   Author: M. Castelan, 10-05-03,
%           Department of Computer Science. 
%           University of York, Heslington, York, YO10 5DD, UK.
%
%   References: 
%
%   [1] R.T.Frankot and Z.Chellappa, "A method for enforcing integrability    
%   in shape from shading algorithms" IEEE Transactions in Pattern
%   Recognition and Machine Intelligence (1988) 10. pp. 439-451
%
%   [2] T.Wei and R.Klette. "Height from gradients with surface curvature and area
%   constraints" (Technical report) CITR-TR-109, Tamaki Campus, University of 
%   Auckland, October 2001. 
%
%   [3] R.Klette,K.Schluns,A.Koschan "Computer Vision: Three-Dimensional Data from Images"  
%   Springer-Verlag Singapore Pte. Ltd.; ISBN: 9813083719
%
%

Y = size(PQ,1);
X = size(PQ,2);

% filtering the gradient fields

if max_pq ~= -1
    for y = 1:Y
        for x = 1:X
            if (abs(PQ(y,x,1)) >  max_pq) && (abs(PQ(y,x,2)) >  max_pq)
               PQ(y,x,:) = 0; 
            end
        end
    end
end


% using the Discrete Cosine Transform to represent Zx and the Discrete Sine Transform DST o represent Zy

tQx=dct(dstn(PQ(:,:,1)')'); 
tQy=dstn(dct(PQ(:,:,2)')');

% here the integration starts

for y = 1:Y 
    AyWy = pi * y / Y; % the differentiation operator is approximated as pi * (u,v)/(M,N)
    for x = 1:X    
        AxWx = pi * x / X;
        
        c(y,x) =(-AxWx*tQx(y,x) - AyWy*tQy(y,x))/(((1+Delta)*(AxWx^2+AyWy^2))+(Miu*(AxWx^2+AyWy^2)^2)); 
        
    end
end

DC = c;
Z = idct2(c);  % Height map delivered by the Inverse Discrete Cosine Transform

Z = Z + max(max(abs(Z))); % normalizing

end

function [Z,Fourier] = ZFrankChelFFT(PQ,max_pq,Delta,Miu)

%
%
%   Global gradient field integration method proposed by 
%   Frankot and Chellappa in [1]
%  
%   Z = ZFrankChel(PQ,max_pq,med_,Delta,Miu) returns the height map Z
%   corresponding to the gradient field PQ, where PQ(:,:,1) =  gradient
%   in X direction and PQ(:,:,2) = gradient in Y direction.
%   max_pq filters the gradients that will be taken into account, it is 
%   almost always set as 4 (As the basis functions to minimize the equation
%   proposed in [1] are the Fourier transform, special care should be taken
%   for values where a high slant is presented - i.e. boundaries -, these ones
%   are set to zero). Delta and Miu are regularizing terms 
%   referring to curvature and area constraints as proposed in [2].
%   when these parameters are both set to zero, the result is the same
%   as originally proposed in [1]
%   
%   A more complete explanation about implementing this method can be found in [3]
%
%   Author: M. Castelan, 10-05-03,
%           Department of Computer Science. 
%           University of York, Heslington, York, YO10 5DD, UK.
%
%   References: 
%
%   [1] R.T.Frankot and Z.Chellappa, "A method for enforcing integrability    
%   in shape from shading algorithms" IEEE Transactions in Pattern
%   Recognition and Machine Intelligence (1988) 10. pp. 439-451
%
%   [2] T.Wei and R.Klette. "Height from gradients with surface curvature and area
%   constraints" (Technical report) CITR-TR-109, Tamaki Campus, University of 
%   Auckland, October 2001. 
%
%   [3] R.Klette,K.Schluns,A.Koschan "Computer Vision: Three-Dimensional Data from Images"  
%   Springer-Verlag Singapore Pte. Ltd.; ISBN: 9813083719
%
%

if max_pq == -1 %  special case when one does not want the gradient field to be filtered.
   max_pq = max(max(PQ));
end   

Y = size(PQ,1);  % the image is supposed to be even, for the antysimetr
X = size(PQ,2);

for y = 1:Y     % filtering the gradients
    for x = 1:X
        if  (abs(PQ(y,x,1)) <=  max_pq) && (abs(PQ(y,x,2)) <=  max_pq)
            P1(y,x) = PQ(y,x,1);
            Q1(y,x) = PQ(y,x,2);
            P2(y,x) = 0;
            Q2(y,x) = 0;            
        else
            P1(y,x) = 0;
            P2(y,x) = 0;
            Q1(y,x) = 0;
            Q2(y,x) = 0;
        end    
    end
end   


FouP = fft2(P1); FouQ = fft2(Q1); % transforming the gradient fields to the frecuency domain
P1 = real(FouP); Q1 = real(FouQ);
P2 = imag(FouP); Q2 = imag(FouQ);

H1 = zeros(Y,X);
H2 = zeros(Y,X);

for y = 1:Y % Here the integration process using the Fourier basis functions starts
    v = y-1;
    
    for x = 1:X
        u = x-1;

        if ~(v == 0 && u == 0)
           
            if u <= ((X/2)-1)
                AxWx = (2*pi*u)/X;
            else
                AxWx = ((2*pi*u)/X)-(2*pi);
            end
            if v <= ((Y/2)-1)
                AyWy = (2*pi*v)/Y;
            else
                AyWy = ((2*pi*v)/Y)-(2*pi);
            end

            %AxWx = sin((2 * pi * u)/(tam));  %this can also be used as an approximation of the differentiation
            %AyWy =  sin((2 * pi * v)/(tam)); %operator for the Fourier transform but only when the range of (u,v)
            %                                  is from -M/2 to (M/2)-1, assuming even M (not this case where 0 <= [u,v] >= M-1) 

            H1(y,x) = ((AxWx*P2(y,x)) + (AyWy*Q2(y,x)))/(((1+Delta)*(AxWx^2+AyWy^2))+(Miu*(AxWx^2+AyWy^2)^2));
            H2(y,x) = ((-AxWx*P1(y,x)) + (-AyWy*Q1(y,x)))/(((1+Delta)*(AxWx^2+AyWy^2))+(Miu*(AxWx^2+AyWy^2)^2));
            
        end
    end
end    

H1(1,1) = 0;
H2(1,1) = 0;

Fourier = complex(H1,H2);

Z = ifft2(Fourier);
Z = -real(Z);             % The real part of the inverse transform contains the height map
Z = Z + max(max(abs(Z))); % normalization
%Z = Z/max(max(Z));

%   This part of the code can be activated if one wants to recover also the new
%   -nearest integrable- gradient field [Zx,Zy] corresponding to the new recovered surface

%for y = 1:Y 
%    v = y-1;
%    for x = 1:X
%        u = x-1;
%        if ~(v == 0 & u == 0)
%            if u <= ((X/2)-1)
%                AxWx = (2*pi*u)/X;
%            else
%                AxWx = ((2*pi*u)/X)-(2*pi);
%            end;
%            if v <= ((Y/2)-1)
%                AyWy = (2*pi*v)/Y;
%            else
%                AyWy = ((2*pi*v)/Y)-(2*pi);
%            end;
%            Zx(y,x) =  -j * AxWx * Fourier(y,x);
%            Zy(y,x) =  -j * AyWy * Fourier(y,x);
%        end;
%    end;
%end;    
%Zx(1,1) = 0;
%Zy(1,1) = 0;

% setting the background as zero if needed

%for y = 1:Y
%    for x = 1:X
%        if PQ(y,x,1) == 0 & PQ(y,x,2) == 0
%           H(y,x) = 0; 
%       end;
%end;
%end;
end



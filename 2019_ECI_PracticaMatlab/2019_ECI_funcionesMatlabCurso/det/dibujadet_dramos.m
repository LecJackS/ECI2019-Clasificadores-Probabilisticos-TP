% [pFR,pFA,EER_DET,DCF_opt,Popt_miss,Popt_fa]=dibujadet_dramos(veroFR,veroFA,color,thick)
% [pFR,pFA,EER_DET,DCF_opt,Popt_miss,Popt_fa]=dibujadet_dramos(veroFR,veroFA,color,thick,percentageMax)
% [pFR,pFA,EER_DET,DCF_opt,Popt_miss,Popt_fa]=dibujadet_dramos(veroFR,veroFA,color,thick,percentageMax,percentageMin)
function  [pFR,pFA,EER_DET,DCF_opt,Popt_miss,Popt_fa]=dibujadet_dramos(veroFR,veroFA,color,thick,percentageMax,percentageMin)

    if (nargin < 6)
        percentageMin=0.0005;
        if (nargin < 5)
            percentageMin=0.0005;
            percentageMax=0.5;
        else
            percentageMax=percentageMax./100;
        end;
    else
        percentageMin=percentageMin./100;
        percentageMax=percentageMax./100;
    end;

    Set_DCF (10, 1, 0.01);
	%[veroFR,veroFA]=optimizar(vector,flags);
    [pFR,pFA] = Compute_DET (veroFR,veroFA);
   
	Pmiss_min = percentageMin+eps;
	Pmiss_max = percentageMax-eps;
	Pfa_min = percentageMin+eps;
	Pfa_max = percentageMax-eps;
	Set_DET_limits(Pmiss_min,Pmiss_max,Pfa_min,Pfa_max);
 
   
   %title ('Curva DET');
   Plot_DET (pFR, pFA, color,thick);
   hold on;
   [DCF_opt Popt_miss Popt_fa] = Min_DCF(pFR,pFA);

   [EER_DET]=getEER_DET(pFA,pFR);
   legend(['EER\_DET: ' num2str(EER_DET) '; DCF\_opt: ' num2str(DCF_opt)])
   % display(['EER_DET: ' num2str(EER_DET)]);
   % display(['DCF_opt: ' num2str(DCF_opt)]);
   %tipo=strcat(color(1),'o');
	%Plot_DET (Popt_miss,Popt_fa,tipo,thick);

  
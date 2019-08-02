function [w,minimumConvergenceCriterium] = train_llr_fusion_convergence_threshold(targets,non_targets,threshold,prior);
%  Train Linear fusion with prior-weighted Logistic Regression objective.
%  The fusion output is encouraged by this objective to be a well-calibrated log-likelihood-ratio.
%  I.E., this is simultaneous fusion and calibration.
%
%  Usage:
%    w = [w,minimumConvergenceCriterium] =
%    train_llr_fusion_convergenge_threshold(targets,non_targets,threshold,prior);
%    w = [w,minimumConvergenceCriterium] =
%    train_llr_fusion_convergenge_threshold(targets,non_targets,threshold);
%   
%  Input parameters:
%    targets:     a [d,nt] matrix of nt target scores for each of d systems to be fused. 
%    non_targets: a [d,nn] matrix of nn non-target scores for each of the d systems.
%    threshold:   a scalar thrshold for the convergence of the CG
%                 algorithm. DEFAULT VALUE 1e-5.
%    prior:       (optional, default = 0.5), a scalar parameter between 0 and 1. 
%                 This weights the objective function, by replacing the effect 
%                 that the proportion nt/(nn+nt) has on the objective.
%                 For general use, omit this parameter (i.e. prior = 0.5).
%                 For NIST SRE, use: prior = effective_prior(0.01,10,1);
%
%  Output parameters:
%    w: a vector of d+1 fusion coefficients. 
%         The first d coefficients are the weights for the d input systems.
%         The last coefficient is an offset (see below).
%    minimumConvergenceCriterium: the minimum of the stopping criterium of
%         the CG algorithm for this case. It only will have utility in
%         cases when convergence is not achieved. For other cases, a value 
%         higher than the threshold will be returned.
%
%  Fusion of new scores:
%    see: LIN_FUSION.m
%                                 
%
%  This code is an adapted version of the m-file 'train_llr_fusion.m' as 
%  included in the FoCal toolkit by Niko Brümmer:
%  www.dsp.sun.ac.za/~nbrummer/focal/index.htm
%
%  Author: Daniel Ramos, November 2006.
disp('nargin')
disp(nargin);
if (nargin<4)
   prior = 0.5;
end;   

if (nargin<3)
   prior = 0.5;
   threshold=1e-5;
end;   

nt = size(targets,2);
nn = size(non_targets,2);
prop = nt/(nn+nt);
weights = [(prior/prop)*ones(1,nt),((1-prior)/(1-prop))*ones(1,nn)];


x = [[targets;ones(1,nt)], -[non_targets;ones(1,nn)]];
%disp(size(x));
w = zeros(size(x,1),1);
%disp(size(w));
offset = logit(prior)*[ones(1,nt),-ones(1,nn)];
%disp(size(offset));
[d,n] = size(x);
old_g = zeros(size(w));

iterations=1000;

minimumConvergenceCriterium=1000000;

warning=1;

for iter = 1:iterations
  old_w = w;
  % s1 = 1-sigma
  %disp(size(w'*x+offset));
  if ((w'*x+offset>100) & (warning==1))
      disp('WARNING: logistic regression training may diverge!.');
      disp('  Possible causes:');
      disp('   -- Input scores are too big! Try score normalization.');
      disp('   -- Training scores are separable.');
      warning=0;
  end;
  s1 = 1./(1+exp(w'*x+offset));
  %disp('x');
  %disp(size(x));
  disp('Tres de g');
  disp('1');
  disp(num2str(x(:,1:5)));
  disp('2');
  disp(num2str(s1(:,1:5)));
  disp('3');
  disp(num2str(weights(:,1:5)));
  g = x*(s1.*weights)';
  %disp('x');
  %disp(size(x));
  %disp('weights');
  %disp(size(weights))

  %disp('s1');
  %disp(size(s1));
  disp('g');
  disp(num2str(g));

  if iter == 1
    u = g;
  else
    disp('u/g/old_g')
    disp('u')
    disp(num2str(u));
    disp('g')
    disp(num2str(g));
    disp('old_g')
    disp(num2str(old_g));
    u = cg_dir(u, g, old_g);
    disp('u after cg_dir')
    disp(num2str(u));
  end
  disp('u:')
  disp(u)
  % line search along u
  ug = u'*g;
  ux = u'*x;
  disp('ug/ux')
  disp(num2str(ug))
  disp(num2str(ux(1:10)))  
  a = weights.*s1.*(1-s1);
  disp('a:')
  disp(num2str(a(1:10)))
  uhu = (ux.^2)*a';
  w = w + (ug/uhu)*u;
  old_g = g;
  disp(['Convergence condition ' num2str(max(abs(w - old_w)))]);
  if max(abs(w - old_w))<minimumConvergenceCriterium
      minimumConvergenceCriterium=max(abs(w - old_w));
  end;
  if max(abs(w - old_w)) < threshold
     disp('w');
     disp(num2str(w));
     disp('old_w');
     disp(num2str(old_w));
     break
  end
end
if iter == iterations
  display(['Convergence not achieved in ' num2str(iterations) ' iterations.']);
end

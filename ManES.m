% @Author: 74006
% @Date:   2018-08-06 14:03:58
% @Last Modified by:   Xiaoyu He
% @Last Modified time: 2020-07-14 17:23:01

% retraction based Manifold ES 

%%%%%%%%%%%%%%%%%%%%%%%%% distribution modeling & sampling %%%%%%%%%%%%%%%%%%%%%%%%%
% each solutions are sampled in the tangent space
%   with 1. a n*p-dimensional isotropic Gaussian projected into the tangent space
%        2. a number m of rank-1 line Gaussian perturbations in the tangent space
% evolution paths are cumulated in the tangent space

function [Ybest, fbest, info] = ManES(problem, Y, options)
timetic = tic;
if ~exist('Y', 'var') || isempty(Y)
    Y = problem.M.rand();
end

[n,p] = size(Y);
dim = problem.M.dim();
fbest = problem.cost(Y);
Ybest = Y;

localdefaults = struct('minsigma',1e-6,'maxiter',inf,...
    'maxcostevals',n*p*100,...
    'sigma0',1,...
    'lambda',4+floor(3*log(dim)),...
    'm','4+floor(3*log(dim))',...
    'damping',1);
options = mergeOptions(mergeOptions(getGlobalDefaults(), localdefaults),options);

lambda = options.lambda;
mu = ceil(lambda/2);
weights = log(mu+1/2)-log(1:mu)';
weights = weights/sum(weights);
mueff = 1/sum(weights.^2);

len = 1: 2*lambda; scum = 0; 
cs = 0.3; 
damping = options.damping; 
% qtarget = 0.1; % differ from SDAES!!!
qtarget = 0.05; % differ from SDAES!!!
sigma_ = options.sigma0;

prevfits = fbest*ones(1, lambda);
     
ccov = 1/(3*sqrt(dim)+5);   
m = eval(options.m);
PCs = zeros(n*p,m);
cc = lambda/dim./4.^(0:m-1);
Yi = zeros(n,p,lambda);
fit_ = zeros(1,lambda);

FEs = 1; iter = 0;
recordGap = 20; recordIter = 1;
recordLength = ceil(options.maxcostevals/lambda/recordGap)+10;
record = zeros(recordLength,5);
record(recordIter,:) = [iter,FEs,fbest,sigma_,toc(timetic)];

while iter < options.maxiter && FEs < options.maxcostevals && sigma_ > options.minsigma
    iter = iter + 1;
    %% sample
    % rank-1 line-Gaussian perturbations in d-dimensional tangent space
    Z_half = PCs * (sqrt(ccov*(1-ccov).^(0:m-1))'.*randn(m,ceil(lambda/2)));
    for i = 1 : ceil(lambda/2)
        % isotropic Gaussian in d-dimensional tangent space 
        %   use projection to get the marginal distribution from a n*p Gaussian 
        t = problem.M.proj(Y,randn(n,p));
        Z_half(:,i) = Z_half(:,i) + sqrt((1-ccov)^m) * t(:);
    end
    Z = [Z_half -Z_half(:,1:floor(lambda/2))];

    %% evaluate and sort
    for i = 1 : lambda
        Yi(:,:,i) = problem.M.retr(Y,reshape(Z(:,i),n,p),sigma_);
        fit_(i) = problem.cost(Yi(:,:,i));
    end
    FEs = FEs + lambda;
    [~,sortedIdx] = sort(fit_);

    %% recombine and move 
    MeanZ = Z(:,sortedIdx(1:mu))*weights;
    Y = problem.M.retr(Y,reshape(MeanZ,n,p),sigma_);

    %% adapt distribution
%     PCs = (1-cc).*PCs + sqrt(cc.*(2-cc)*mueff).*MeanZ;
    PCs = (1-cc).*PCs + real(sqrt(cc.*(2-cc)*mueff)).*MeanZ;
    % move to new tangent space
    for i = 1 : m
        pc = problem.M.transp([],Y,reshape(PCs(:,i),n,p));
        PCs(:,i) = pc(:);
    end
    % adapt step size
    [~, imax] = sort([prevfits, fit_]);
    R1 = sum(len(imax<=lambda));
    U1 = R1 - lambda*(lambda+1)/2;
    W = U1 - lambda^2/2;
    W = W / sqrt(lambda^2*(2*lambda+1)/12);
    scum = (1-cs) *scum + sqrt(cs*(2-cs))*W;
    sigma_ = sigma_ * exp((normcdf(scum)/(1-qtarget)-1)/damping); 
    prevfits = fit_;

    %% trace elite
    if fit_(sortedIdx(1)) < fbest
        fbest = fit_(sortedIdx(1));
        Ybest = Yi(:,:,sortedIdx(1));
    end

    %% save records
    if mod(iter,recordGap)==1
        if options.verbosity > 0
            fprintf('[#%d FEs:%d] fit = %+.16e sigma = %g\n',iter,FEs,fbest,sigma_);
        end
        record(recordIter,:) = [iter,FEs,fbest,sigma_,toc(timetic)];
        recordIter = recordIter + 1;
    end
end
if iter > record(recordIter-1,1)
    record(recordIter,:) = [iter,FEs,fbest,sigma_,toc(timetic)];
    recordIter = recordIter + 1;
end
record(recordIter:end,:) = [];
info = array2table(record, 'VariableNames', {'iter','costevals','cost','sigma','time'});
info = table2struct(info);
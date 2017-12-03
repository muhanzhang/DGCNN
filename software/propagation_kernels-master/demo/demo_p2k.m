%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %   DEMO_P2K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% LOAD demo data
load('BZR_mat');    

num_nodes = size(A, 1);                     % total number of nodes
num_graphs = size(unique(graph_ind),1);     % number of graphs
num_dims = size(attributes,2);              % number of attribute dimensions

% standardize attributes 
attributes = zscore(attributes);

% % statistical whitening (-> transform data to have identiy covaraince) 
% mu = mean(attributes,1);
% attributes = bsxfun(@minus,attributes, mu);
% SIGMA = cov(attributes);
% [U,S,~] = svd(SIGMA);
% B = U*S^(-1/2);
% attributes = attributes*B;


USE_DEGREES = false;                    % use node degrees (instead of node labels) for structure propagation
if USE_DEGREES
    degs = full(sum(A,2));
    labels = degs;
end
[~,~,labels] = unique(labels);          % handle zero node labels

% row-normalize A
row_sum = sum(A, 2);
row_sum(row_sum==0)=1;                  % avoid dividing by zero => disconnected nodes
A = bsxfun(@times, A, 1 ./ row_sum);

% PK PARAMETERS 
num_iter = 10;      % number of kernel iterations to test
w = 1e-5;
w_attr = 1; 
dist = 'tv';        % 'tv' or 'hellinger'         
dist_attr = 'l1';   % 'l1' or 'l2'


% PROPAGATE ATTRIBUTES or USE SIMPLE HASH 
USE_ATTR_PROP = true;


% ========================================================================
% PREPARE INPUT DATA
distribution_features = accumarray([(1:num_nodes)', labels], 1, [num_nodes, max(labels)]);

% TRANSFORMATION FOR LABELS
transformation = @(features) label_diffusion(features, A);

% TRANSFORMATION FOR ATTRIBUTES
if USE_ATTR_PROP
    fprintf('...initializing GMs\n')
    runtime = cputime;
    
    num_samples = 100;              % number of samples to evaluate Gaussiam Mixtures
    
    % GMM initialization: parameters of mixture components (one for every node)
    mus = attributes;                               % mixture means
    Ks  = zeros(num_dims, num_dims, num_nodes);     % mixture covariance 
    Ks_pre = cov(attributes) + 0.01 * eye(num_dims);
    % Ks_pre = diag(var(attributes));
    % Ks_pre = marginal_std^2 * eye(num_dims);    % same variance across dimensions
    for i = 1:num_nodes
        Ks(:, :, i) = Ks_pre;
    end
    
    evaluate_pdfs();            % clear persistent variables in evaluate_pdfs (only needs to happen if mus, Ks, or x changes)
    
    rng(0);                     % initialize random seed for reproducible results
    % create handle to tranformation for propagation_kernel
    rand_samples = randperm(num_nodes, num_samples)';
    rand_samples = attributes(rand_samples,:);
    transformation_attr = @(attributes) gmm_propagation(A, mus, Ks, rand_samples);  % CAUTION: gmm_propagation uses persistent variables
    runtime = cputime-runtime; 
    fprintf('initialization time: %2.1f'''' (%2.1f'')\n', runtime, runtime/60)
end
    

run = 1;
rng(run);   % initialize random seed for reproducible results - should be varied for experiemnts
runtime = cputime;
start = tic;
% ========================================================================
% COMPUTE kernel
fprintf('...computing propagation kernel\n')
if USE_ATTR_PROP
    gmm_propagation();  % clear persistent variables in gmm_propagation (should happen before every call to propagation_kernel)
    initial_attributes = transformation_attr([]); % get initial attribute distributions
    K = propagation_kernel(distribution_features, graph_ind, transformation, num_iter, ...
                           'w', w,'distance', dist, ...
                           'attr', initial_attributes, 'w_attr', w_attr, 'dist_attr', dist_attr,...
                           'trans_attr', transformation_attr);
else % do NOT use attribute propagation  
    K = propagation_kernel(distribution_features, graph_ind, transformation, num_iter, ...
                           'w', w,'distance', dist, ...
                           'attr', attributes, 'w_attr', w_attr, 'dist_attr', dist_attr);
end

runtime = cputime-runtime; 
fprintf('cputime for kernel computation: %2.1f'''' (%2.1f'')\n', runtime, runtime/60)
toc(start)


% % ========================================================================
% % KERNEL EVALUATION (libSVM Classification)
% % CAUTION: svmtrain.m in libSVM needs to be renamed to svmtrain_libsvm.m
% fprintf('...svm evaluation\n')
% addpath(genpath('/path/to/libsvm/'))
% svm_options = @(c)(['-q -t 4 -c ' num2str(c)]);
% 
% num_folds = 10; 
% cost = 1;
% rng(run);
% c = cvpartition(numel(graph_labels),'kfold',num_folds); % generate data splits
% 
% accurracies = zeros(num_folds, 1);
% for i=1:num_folds
%     train_ind = find(training(c,i)==1);
%     test_ind = find(training(c,i)==0);  
% 
%     K_train = [(1:length(train_ind))' K(train_ind,train_ind)];
%     K_test = [(1:length(test_ind))' K(test_ind,train_ind)];
% 
%     % SVM prediciton
%     model = svmtrain_libsvm(graph_labels(train_ind),K_train, svm_options(cost));    
%     [y_pred, acc, ~] = svmpredict(graph_labels(test_ind),K_test, model, '-q');
%     accurracies(i) = acc(1);
% end
% fprintf('%d-fold CV accuracy (+/- stdv) = %2.2f (+/- %0.2f) \n',num_folds,mean(accurracies),std(accurracies))                     




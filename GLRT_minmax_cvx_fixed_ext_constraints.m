%%% ya Ali el walleeeeeee

tic

clear all ;
close all;
clc; 

%rng(456987);
rng(12121)
N = 10 ; % length of sequence 
q = 30 ;  % number of unknown channel parameters 
M = 15 ;  % number of hypotheses 

%%%% BUILD super labels [THIS IS EASY TO CHANGE]
%%%% FUNCTIONS TO CHANGE IF THIS IS CHANGED ARE: 
%%%% super_lbl_from_lbl_journal.m
%%%% S_from_lbl_journal.m
%%%% S_comp_star_from_lbl_journal.m
% S1 = [1:3] ;
% S2 = [4:5] ;
% S3 = [6];
% S4 = [7:8];
% S5 = [9:10];
% S6 = [11];
% S7 = [12];
% S8 = [13:15];

%%%% BUILD mapper
% if ismember(lbl, S1)
%     super_lbl = 1 ;
% elseif ismember(lbl, S2)
%     super_lbl = 2 ;
% elseif    ismember(lbl, S3)
%     super_lbl = 3 ;
% end
   

% new mapping: 
S1 = [1:4];
S2 = [5:10];
S3 = [11:12];
S4 = [13:15];




NN = N+q-1 ; % length of measurements 

%% Generate   the code book of sequence of symbols

%%% 2-PAM
rng(12121)

%V_org = ( randi([0,1],N,M) - 0.5 ) * 1 ;
V_org = load('fixed_data/V_org.mat'); V_org = V_org.V_org;

%V_org = randn(N,M) ; 

%%% 4-PAM
%V_org = ( randi([0 3],N,M) * (1/3) ) - .5 ;

%%% randn
%V_org = 0 + 1.*randn(N,M);
%a = -.5 ; b = .5 ;
%V_org = a + (b-a).* rand(N,M) ;

% create toeplitz matrix for each possibility i \in [M]
VV_org = zeros(NN,q,M) ;
for i = 1:M
    VV_org(:,:,i) =   Build_H(V_org(:,i)',q);  
end


% GOAL: detect which sequence (of length N) is sent from M the
% possibilities with an unknown channel parameters theta_oracle
% theta_oracle = 0 + (1/1)*randn(q,1) ;
% theta_oracle = randi([1 100], q,1);
% theta_oracle = theta_oracle / max(theta_oracle) ;
% theta_oracle = sort(theta_oracle,'descend');

%% beginning of transmission 

%%% ini
trails = 1000 ;
cnt  = 0 ;
cnt_sup = 0 ;

% this is for cvx optimization
c = 2.15;
p0 = 2 ;

%%%% FOR SAVING SHIT 
y            = zeros(NN, trails);
y_pert       = zeros(NN, trails);
eta_star     = zeros(NN, trails);

tru_lbl      = load('fixed_data/tru_lbls.mat'); 
tru_lbl = tru_lbl.tru_lbl;

est_lbl      = zeros(trails , 1);
est_sup_lbl  = zeros(trails , 1);
tru_sup_lbl  = zeros(trails , 1);
est_lbl_pert = zeros(trails , 1);
est_sup_lbl_pert = zeros(trails , 1);
D_2 = zeros(trails , 1);
D_inf = zeros(trails , 1);

noise = zeros(NN,trails);

theta_oracle = zeros(q,trails);



for tr = 1:trails

%tru_lbl(tr) = randi([1 M],1,1);

tru_sup_lbl(tr) = super_lbl_from_lbl_journal(tru_lbl(tr));

%theta_oracle(:,tr) = 0 + (1/1)*randn(q,1) ;

theta_oracle = load('fixed_data/theta_oracle.mat'); 
theta_oracle = theta_oracle.theta_oracle;

v_t = V_org(:,tru_lbl(tr)) ; 

% get corresponding toeplitz matrix V of v_t

VV_t = Build_H(v_t',length(theta_oracle(:,tr))) ; 

%noise(:,tr) = 0 + .5.*randn(NN,1);
noise = load('fixed_data/noise.mat'); 
noise = noise.noise;

%% at Rx
y(:,tr) = VV_t*theta_oracle(:,tr) + noise(:,tr) ; 

%plot(y)

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% MCHT
%%%%%%%%%%%%%%%%%%%%%%%%%%

[est_lbl(tr),J] = MCHT_mod(y(:,tr),VV_org) ;
est_sup_lbl(tr) = super_lbl_from_lbl_journal(est_lbl(tr));

    % get S_comp_star (basically just implement argmin for j \in S') (method 2)
    QQ = 3 ;
    S_comp_star_from_lbl_ = S_comp_star_from_lbl_journal(tru_lbl(tr),...
                            J, QQ); 
    
    % get grad_cvx for
    grad_cvx = zeros(M,NN);
    for i = 1:M
        V_att = VV_org(:,:,i);
        gamma = V_att*pinv(V_att'*V_att)*V_att' ;
        grad_cvx(i,:) = 2.*( eye(NN) - gamma )*( y(:,tr) - gamma*y(:,tr) ) ;    
    end
                        
                        
    % get S_T(k)
    SS = S_from_lbl_journal(tru_lbl(tr)) ;
        
    %%% jj = J_hat
    jj = S_comp_star_from_lbl_ ;
    
    
    
    
    cvx_begin quiet
    variables eta_best_cvx(NN)
        %minimize(     norm(eta_best_cvx,p0)       )
    minimize(    square_pos( norm(eta_best_cvx) ) )
    subject to
        % constraints
        for ll = SS
            for j = jj
            eta_best_cvx' * ( grad_cvx(j,:)' - grad_cvx(ll,:)')  <=...
                J(ll) - J(j) - c
            end
        end   
    cvx_end
        
    
         % check super lbls here
         eta_star(:,tr) =    eta_best_cvx ;
         y_pert(:,tr)   =    y(:,tr) + eta_best_cvx ;
        
         % MCHT with pert
        [est_lbl_pert(tr),J_pert] = MCHT_mod(y_pert(:,tr),VV_org) ;
        
        % rho_2
        D_2(tr) = ( norm(eta_best_cvx,2) / norm(y(:,tr),2)  ) ;
        
        % rho_inf
        D_inf(tr) = ( norm(eta_best_cvx,'inf') / norm(y(:,tr),'inf')  ) ;
        
        est_sup_lbl_pert(tr) = super_lbl_from_lbl_journal(est_lbl_pert(tr)) ;

        %%%% CA_pert = 100 - below
        if est_lbl_pert(tr) ~= tru_lbl(tr) 
            cnt = cnt+1 ;
            
        end
        
        
        %%%% CA_sup_pert = 100 - below
        if est_sup_lbl_pert(tr) ~= tru_sup_lbl(tr) 
            cnt_sup = cnt_sup+1 ;
            
        end


end

%cnt

CA = 81.10 % this is always fixed 

%%%% CA_sup
% we are counting the tru_sup_lbl and est_sup_lbl

cnt_CA_sup = 0 ;
for i = 1:trails
        if est_sup_lbl(i) == tru_sup_lbl(i) 
            cnt_CA_sup = cnt_CA_sup+1 ;
            
        end
end

CA_sup = 100* (       cnt_CA_sup / trails    )


CAp = 100*(trails-cnt) / trails 

CA_sup_pert = 100*(trails-cnt_sup) / trails

fooling_ratio = (CA-CAp)/CA 

ELA = ( 100 - CAp ) / CA 

fooling_ratio_sup = (CA_sup-CA_sup_pert)/CA_sup 

ELA_sup = ( 100 - CA_sup_pert ) / CA_sup 


rho_2 = mean(D_2)

rho_i = mean(D_inf)

toc



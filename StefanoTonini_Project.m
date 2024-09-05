clc 
clear all
close all 
%% Rotary Inverted Pendulum

% Parameters
mp = 0.1;      %The mass of the pendulum [kg]
Lp = 0.3;      %length of the pendulum [m]  
lp = 0.15;     %The position of the center of mass of the pendulum [m]
Jp = 0.003;    %The moment of inertia of the pendulum [kgm^2]
Lr = 0.15;     %The length of the rotary arm [m]  
Jr = 0.01;     %The moment of inertia of the rotary arm [kgm^2] 
Bp = 0.4;      %The viscous friction coefficients of the pendulum [Nms/rad] 
Br = 0.2;      %The viscous friction coefficients of rotary arm [Nms/rad] 
g = 9.81;      %The gravity constant [m/s^2]

%% Question 1: Linearization and check if system is AS, stable or unstable

% -- Simulation settings
simT = 0:0.01:6; %integrate over this time Time Domain
q0   = [0.05; 0; 0.06; 0]; %Initial Conditions

% Equilibrium point

theta_eq = 0;
alpha_eq = 0;
theta_dot_eq = 0;
alpha_dot_eq = 0;


% Define symbolic variables
syms theta alpha theta_dot alpha_dot u real;

% Define the state vector and input
x = [theta; alpha; theta_dot; alpha_dot];
%u = sym('u', [1 1], 'real');


% M(q)
M = [Jr + mp*(Lr^2 + lp^2*(1-cos(alpha))^2), mp*lp*Lr*cos(alpha);
     mp*lp*Lr*cos(alpha), Jp + mp*lp^2];

% C(q, q_dot)
C = [2*mp*lp^2*alpha_dot*sin(alpha)*cos(alpha), -mp*lp*Lr*alpha_dot*sin(alpha);
     -mp*lp^2*theta_dot*sin(alpha)*cos(alpha), 0];
disp('numeric M = ');
disp(M);
disp('numeric C = ');
disp(C);
% f_v(q_dot)
fv = [Br*theta_dot;
      Bp*alpha_dot];

% G(q)
G = [0;
     -mp*lp*g*sin(alpha)];

% tau
tau = [u; 0];

% Define the equations of motion
q = [theta; alpha];
q_dot = [theta_dot; alpha_dot];
q_ddot = M \ (tau - C*q_dot - fv - G);

% Define the state-space representation
f = [q_dot;
     q_ddot];

% Linearize at equilibria

A = double(subs(jacobian(f, x), {theta, alpha, theta_dot, alpha_dot, u}, {theta_eq, alpha_eq, theta_dot_eq, alpha_dot_eq, 0}));
B = double(subs(jacobian(f, u), {theta, alpha, theta_dot, alpha_dot, u}, {theta_eq, alpha_eq, theta_dot_eq, alpha_dot_eq, 0}));

% Display the results
disp('Linearized state-space representation:');
disp('A = ');
disp(A);
disp('B = ');
disp(B);

% Check stability and controllability
eigA = eig(A);
disp('Eigenvalues of A:');
disp(eigA);

co = ctrb(A, B);
controllability_rank = rank(co);

disp(['Controllability Matrix Rank: ', num2str(controllability_rank)]);

% -- Plant dimensions
[n,~] = size(A); %[n x n]
[~,p] = size(B); %[n x p]


% Stability: Eigenvalues Test
ev = real(eig(A)); %real part of the eigenvalues of A

disp('--------------------------------------------------------------------------')
disp('Internal Stability Test (open-loop)')
disp('*Eigenvalues test:')
if all(ev<0)
    disp(['    ' 'all eigenvalues have negative real part --> system is stable'])
else
    disp(['    ' 'at least one eigenvalue has positive real part --> system is unstable'])
end
disp('*Open-loop eigenvalues real parts: ')
disp(['    ', mat2str(ev')]);

% Controllability Test
ctr = rank(ctrb(A,B)); %controllability matrix

disp('--------------------------------------------------------------------------')
disp('Controllability Test');
disp('*Controllability matrix test:');
if ctr==n
    disp(['    ' 'ctrb matrix is full rank --> system is controllable'])
else
    disp(['    ' 'ctrb matrix is rank deficient --> system is NOT controllable'])
    disp('*plant dimension:');
    disp(['    ', num2str(n)]);
    disp('*ctrb matrix rank:');
    disp(['    ', num2str(ctr), newline]);
end

% Checking asymptotic stability of the system using LMIs

yalmip('clear'); %clear the memory of the solver

% -- Define settings to give to the solver
opts = sdpsettings('solver', 'mosek', 'verbose', 0);


% -- Define variables to be used in the optimization problem
P = sdpvar(n,n);
fake_zero = 1e-3;

% -- Define LMIs are constraints of the optmization problem
constr = [P >= fake_zero*eye(n);
          A'*P + P*A <= -fake_zero*eye(n)];
      
% -- Solve the opmtimization problem
sol = solvesdp(constr, [], opts);

% -- Extract the numerical values of P
P_value = value(P);

% -- Check that the problem was successfully solved
[primal_res, dual_res] = check(constr); %residuals of the constraints

% -- All the residuals must be greater than zero. Sometimes you can accept
% slightly negative residuals
feas = all([primal_res; dual_res] > -fake_zero);

disp(newline);
disp('-----------------------')
disp('Checking asymptotic stability using the LMI condition');
if( feas )
    disp('The LMI is feasible, the system is AS');
else
    disp('The LMI is infeasible, the system is NOT AS');
end




%% Question 2: Gain Matrices

%% Feedback design: desired Convergence Rate
% -- Clear the internal memory of YALMIP
yalmip('clear')

% -- Optimization variables
W = sdpvar(n,n);
X = sdpvar(p,n);

% -- Constraints
alpha_b = 2;
constr = [ W >= fake_zero*eye(n)                ;...
    (A*W+B*X)+(A*W+B*X)' <= -2*alpha_b*W ];

% -- Solve the problem
sol = solvesdp(constr,[],opts);

% -- extract the results
W_b = value(W);
X_b = value(X);
K_b = X_b*inv(W_b);

% -- Check constraints
[primalfeas,dualfeas] = check(constr);
disp('--------------------------------------------------------------------------')
disp(['LMI Feedback design (alpha = ' num2str(alpha_b) ')'])
disp('*Constraints check:')
if all([primalfeas;dualfeas] > -fake_zero)
    disp(['    ', 'Constraints are satisfied --> system is stable'])
    disp('*Gain matrix K:');
    disp(['    ', num2str(K_b)]);
else
    disp(['    ', 'Constraints are NOT satisfied --> system is unstable'])
end
disp('*Closed-loop eigenvalues real parts:')
disp(['    ' num2str(real(eig(A+B*K_b)'))]);

% -- Simulation
[simT_b,simQ_b]  = ode45(@(t,q)f_NLDyna(q, Jr, mp, Lr, lp, Jp, Br, Bp, g, K_b*q),simT,q0);

%% Feedback design: desired Convergence Rate and minimum control effort
% -- Clear the internal memory of YALMIP
yalmip('clear')

% -- Optimization variables
W = sdpvar(n,n);
X = sdpvar(p,n);
k = sdpvar(1,1);

% -- Constraints
alpha_c = 2;
constr = [ W >= eye(n)               ;...
    (A*W+B*X)+(A*W+B*X)' <= -2*alpha_c*W ;...
    [ k*eye(n)      X'   ;...
    X      k*eye(p) ]   >= fake_zero*eye(n+p)];

% -- Solve the problem
sol = solvesdp(constr,k,opts);

% -- Extract the results
W_c = value(W);
X_c = value(X);
k_c = value(k);
K_c = X_c*inv(W_c);

% -- Check constraints
[primalfeas,dualfeas] = check(constr);
disp('--------------------------------------------------------------------------')
disp(['LMI Feedback design (alpha = ' num2str(alpha_c) ', minimize |K|)' ])
disp('*Constraints check:')
if all([primalfeas;dualfeas] > -fake_zero)
    disp(['    ' 'Constraints are satisfied --> system is stable'])
    disp('*Gain matrix K:');
    disp(['    ' num2str(K_c)]);
    disp(['*Minimum bound on gain matrix norm: |K| < ', num2str(k_c)]);
else
    disp(['    ', 'Constraints are NOT satisfied --> system is unstable'])
end
disp('Closed-loop eigenvalues real parts:')
disp(['    ', num2str(real(eig(A+B*K_c)'))]);

P_c = inv(W_c);
M_c = sqrt(k_c)

% -- Simulation
[simT_c,simQ_c]  = ode45(@(t,q)f_NLDyna(q, Jr, mp, Lr, lp, Jp, Br, Bp, g, K_c*q),simT,q0);

%% Speed of convergence alpha_bar estimation

yalmip('clear');

P = sdpvar(n,n);

% -- To find alpha_bar we start with a guess value
alpha_guess = 1e-3;

% -- Check if the LMI is feasible for this choice of alpha
constr = [P >= fake_zero*eye(n);
    (A+B*K_b)'*P + P*(A+B*K_b) <= -2*alpha_guess*P];

sol = solvesdp(constr, [], opts);

[primal_res, dual_res] = check(constr);
feas = all( [primal_res; dual_res] > -fake_zero );

% -- If the LMI is feasible, double the value of alpha_guess and test again
% if the LMI is feasible
while( feas )
    alpha_guess = alpha_guess*2;

    constr = [P >= fake_zero*eye(n);
        (A+B*K_b)'*P + P*(A+B*K_b) <= -2*alpha_guess*P];

    sol = solvesdp(constr, [], opts);
    [primal_res, dual_res] = check(constr);
    feas = all( [primal_res; dual_res] > -fake_zero );

end

% -- The final value of alpha_guess makes the LMI infeasible, and thus it
% is the upper bound on alpha

%problem is infeasible for alpha > alpha_upper
alpha_upper = alpha_guess;
%problem is feasible for alpha < alpha_lower
alpha_lower = alpha_guess/2;



% -- Bisection method to find estimate of alpha_bar

threshold = 1e-3; %threshold used to stop the bisection procedure

while( (alpha_upper - alpha_lower) > threshold ) %stopping condition

    % -- Test if LMI is feasible for alpha_guess equals to the middle point
    % between alpha_lower and alpha_upper
    alpha_guess = (alpha_upper + alpha_lower)/2;

    constr = [P >= fake_zero*eye(n);
        (A+B*K_b)'*P + P*(A+B*K_b) <= -2*alpha_guess*P];

    sol = solvesdp(constr, [], opts);
    [primal_res, dual_res] = check(constr);
    feas = all( [primal_res; dual_res] > 0 );

    if( feas )
        % -- Problem is feasible for all alpha < alpha_guess, increase the
        % lower bound alpha
        alpha_lower = alpha_guess;
    else
        % -- Problem is infeasible for all alpha > alpha_guess, lower the
        % upper bound on alpha
        alpha_upper = alpha_guess;
    end
end

% -- The estimate of alpha_bar is the final value of alpha_lower
alpha_bar = alpha_lower;
disp('-----------------------')
disp( ['Estimated value of alpha_bar: ', mat2str(alpha_bar)] );
disp('-----------------------')


%% Question 3: Estimation of the overshoot Mb

yalmip('clear');

% -- variables
P = sdpvar(n,n);
k = sdpvar(1,1);

% -- LMIs
constr = [ eye(n) <= P <= k*eye(n);
    (A+B*K_b)'*P + P*(A+B*K_b) <= -2*alpha_bar*P ];

sol = solvesdp(constr, k, opts);

[primal_res, dual_res] = check(constr);

feas = all([primal_res; dual_res] > -fake_zero);


disp(newline);
disp('-----------------------')
disp('Estimating overshoot M');

k_val = value(k);
M = sqrt(k_val);

if(feas)
    disp( ['The problem is feasible, value of M: ', mat2str(M)] );
    disp('-----------------------')
else
    disp('The problem is infeasible');
    disp('-----------------------')
end

P_b = inv(W_b);
M_b = sqrt(max(eig(P_b))/min(eig(P_b)))
%% Speed of convergence alpha_bar estimation

yalmip('clear');

P = sdpvar(n,n);

% -- To find alpha_bar we start with a guess value
alpha_guess = 1e-3;

% -- Check if the LMI is feasible for this choice of alpha
constr = [P >= fake_zero*eye(n);
    (A+B*K_b)'*P + P*(A+B*K_b) <= -2*alpha_guess*P];

sol = solvesdp(constr, [], opts);

[primal_res, dual_res] = check(constr);
feas = all( [primal_res; dual_res] > -fake_zero );

% -- If the LMI is feasible, double the value of alpha_guess and test again
% if the LMI is feasible
while( feas )
    alpha_guess = alpha_guess*2;

    constr = [P >= fake_zero*eye(n);
        (A+B*K_b)'*P + P*(A+B*K_b) <= -2*alpha_guess*P];

    sol = solvesdp(constr, [], opts);
    [primal_res, dual_res] = check(constr);
    feas = all( [primal_res; dual_res] > -fake_zero );

end

% -- The final value of alpha_guess makes the LMI infeasible, and thus it
% is the upper bound on alpha

%problem is infeasible for alpha > alpha_upper
alpha_upper = alpha_guess;
%problem is feasible for alpha < alpha_lower
alpha_lower = alpha_guess/2;

% -- Bisection method to find estimate of alpha_bar

threshold = 1e-3; %threshold used to stop the bisection procedure

while( (alpha_upper - alpha_lower) > threshold ) %stopping condition

    % -- Test if LMI is feasible for alpha_guess equals to the middle point
    % between alpha_lower and alpha_upper
    alpha_guess = (alpha_upper + alpha_lower)/2;

    constr = [P >= fake_zero*eye(n);
        (A+B*K_b)'*P + P*(A+B*K_b) <= -2*alpha_guess*P];

    sol = solvesdp(constr, [], opts);
    [primal_res, dual_res] = check(constr);
    feas = all( [primal_res; dual_res] > 0 );

    if( feas )
        % -- Problem is feasible for all alpha < alpha_guess, increase the
        % lower bound alpha
        alpha_lower = alpha_guess;
    else
        % -- Problem is infeasible for all alpha > alpha_guess, lower the
        % upper bound on alpha
        alpha_upper = alpha_guess;
    end
end

% -- The estimate of alpha_bar is the final value of alpha_lower
alpha_bar_2 = alpha_lower;
disp('-----------------------')
disp( ['Estimated value of alpha_bar: ', mat2str(alpha_bar_2)] );
disp('-----------------------')

%% Estimation of the overshoot Mc

yalmip('clear');

% -- variables
P = sdpvar(n,n);
k = sdpvar(1,1);

% -- LMIs
constr = [ eye(n) <= P <= k*eye(n);
    (A+B*K_c)'*P + P*(A+B*K_c) <= -2*alpha_bar*P ];

sol = solvesdp(constr, k, opts);

[primal_res, dual_res] = check(constr);

feas = all([primal_res; dual_res] > -fake_zero);

disp(newline);
disp('-----------------------')
disp('Estimating overshoot M');

k_val = value(k);
M = sqrt(k_val);

if(feas)
    disp( ['The problem is feasible, value of M: ', mat2str(M)] );
else
    disp('The problem is infeasible');
    disp('-----------------------')
end

% Compare solutions
% -- State norm evolution over time

normX_b = vecnorm(simQ_b');
normX_c = vecnorm(simQ_c');
% -- Input evolution over time



x_bound_b = M_b*norm(q0)*exp(-alpha_bar*simT);
x_bound_c = M_c*norm(q0)*exp(-alpha_bar_2*simT);

u_b = (K_b*simQ_b')';
u_c = (K_c*simQ_c')';


%% Plot
% Plot 1
figure(1)
subplot(211)
plot(simT_b,normX_b, 'Color',[0 .3470 .8410], 'LineWidth',1.5)
hold on
plot(simT_c,normX_c, 'Color',[.9290 .4940 .1250], 'LineWidth',1.5)
grid on
grid minor
legend('$\alpha$ only','minimize $|K|$', 'interpreter', 'latex')
title('State Norm')
xlabel('$t[s]$', 'interpreter', 'latex')
subplot(212)
plot(simT_b,u_b, 'Color',[0 .3470 .8410], 'LineWidth',1.5)
hold on
plot(simT_c,u_c, 'Color',[.9290 .4940 .1250], 'LineWidth',1.5)
grid on
grid minor
legend('$\alpha$ only','minimize $|K|$', 'interpreter', 'latex')
title('Control Input')
xlabel('$t[s]$', 'interpreter', 'latex')

% Plot 2
figure(2)
subplot(211)

plot(simT_b,simQ_b(:,1), 'Color',[0 .3470 .8410], 'LineWidth',1.5)
hold on
plot(simT_c,simQ_c(:,1), 'Color',[.9290 .4940 .1250], 'LineWidth',1.5)
grid on
grid minor
legend('$\alpha$ only','minimize $|K|$', 'interpreter', 'latex')
title('theta angle')
xlabel('$t[s]$', 'interpreter', 'latex')

subplot(212)

plot(simT_b,simQ_b(:,3), 'Color',[0 .3470 .8410], 'LineWidth',1.5)
hold on
plot(simT_c,simQ_c(:,3), 'Color',[.9290 .4940 .1250], 'LineWidth',1.5)
grid on
grid minor
legend('$\alpha$ only','minimize $|K|$', 'interpreter', 'latex')
title('alpha angle')
xlabel('$t[s]$', 'interpreter', 'latex')


% Plot 3
figure(3), clf
subplot(211)
hold on
grid minor
plot(simT_b,normX_b, 'Color',[0 .3470 .8410], 'LineWidth',1.5)
plot(simT_b, x_bound_b, 'Color',[.9290 .4940 .1250]);
legend('$|x(t)|$', '$M|x(0)|e^{-\bar \alpha t}$', 'interpreter', 'latex');
title('Trajectory');
xlabel('$t[s]$', 'interpreter', 'latex');

subplot(212)
hold on
grid minor
plot(simT_c,normX_c, 'Color',[0 .3470 .8410], 'LineWidth',1.5)
plot(simT_c, x_bound_c, 'Color',[.9290 .4940 .1250]);
legend('$|x(t)|$', '$M|x(0)|e^{-\bar \alpha t}$', 'interpreter', 'latex');
title('Trajectory');
xlabel('$t[s]$', 'interpreter', 'latex');
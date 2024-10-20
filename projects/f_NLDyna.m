function dq = f_NLDyna(q, Jr, mp, Lr, Lp, Jp, Br, Bp, g, u)
    % This function contains the Non-Linear Equations describing the System Dynamics
    % state: q = [theta_r, theta_r_dot, alpha, alpha_dot]

    % Extract the states
    theta = q(1);
    theta_dot = q(3);
    alpha = q(2);
    alpha_dot = q(4);

    % Precompute common terms
    sin_alpha = sin(alpha);
    cos_alpha = cos(alpha);
    sin_2alpha = sin(2 * alpha);
    cos_2alpha = cos(2 * alpha);

    %  M(q)
    M11 = Jr + mp * (Lr^2 + Lp^2 * (1 - cos_alpha^2));
    M12 = mp * Lp * Lr * cos_alpha;
    M21 = M12;
    M22 = Jp + mp * Lp^2;

    M = [M11, M12; M21, M22];

    %  C(q, q_dot)
    C11 = 2 * mp * Lp^2 * alpha_dot * sin_alpha * cos_alpha;
    C12 = -mp * Lp * Lr * alpha_dot * sin_alpha;
    C21 = -mp * Lp^2 * theta_dot * sin_alpha * cos_alpha;
    C22 = 0;

    C = [C11, C12; C21, C22];

    % Friction vector f_v(q_dot)
    f_v = [Br * theta_dot; Bp * alpha_dot];

    % Gravity vector G(q)
    G = [0; -mp * g * Lp * sin_alpha];

    % Input torque tau
    tau = [u; 0];

    % State derivatives
    q_dot = [theta_dot; alpha_dot];

    % Compute dq using the dynamics equation: M(q)*q_ddot + C(q,q_dot)*q_dot + f_v(q_dot) + G(q) = tau
    q_ddot = inv(M) * (tau - C * q_dot - f_v - G);

    % Concatenate q_dot and q_ddot to form the state derivative dq
    dq = [q_dot; q_ddot];
end

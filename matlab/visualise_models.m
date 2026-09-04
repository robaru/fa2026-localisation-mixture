% visualise_models.m
%
% Shows how the two localization models produce response distributions
% on a circle (polar angle space).
%
% MODEL 1 - Original mixture:
%   p_simple(phi | phi_t) = w * VM(phi_t, k) + (1-w) * VM(180-phi_t, k)
%
% MODEL 2 - With horizontal prior:
%   same as above but multiplied by a prior that favours 0 deg (front)
%   and 180 deg (back), i.e. the horizontal plane.
%   p_full(phi | phi_t) = p_simple(phi | phi_t) * (.5 * VM(0, k_prior) + .5 * VM(180, k_prior))
%   
% phi   = response angle (polar, degrees, 0 = front horizontal, 180 = back)
% phi_t = target angle (the true sound polar angle)
% w     = probability of responding in the correct hemifield (1 = perfect)
%           (1-w gives you the probability of doing a quadrant error)
% the vonMises distribution (VM) uses concetration (k) instead of variance
% (as for gaussians). We will use a simple conversion (kappa ~ 1 /
% sigma_rad^2 - breaks down for very large sigma, but fine here). 
% The remaining parameters are: 
% sigma = spread of responses in degrees (higher = more uncertain - our polar uncertainty)
% sigma_prior = how strongly the prior pulls responses toward the horizontal
%               plane (lower = stronger pull, higher = weaker / flatter prior)


clear; close all;

%% ---- Parameters to play with -----------------------------------------------

phi_t    = 45;   % target elevation in degrees (try 30, 45, 60 ...)
phi_resp = 60;   % example response to evaluate (try values near phi_t, or near 135 for a confusion)

% Three example listeners (vary sigma and w to see different behaviours)
listeners = {
%   sigma(deg)   w      label
    15,          0.9,   "Precise listener";
    40,          0.75,  "Typical listener";
    70,          0.55,  "Imprecise listener";
};

% Prior strength examples (for Model 2, right column of the figure)
sigma_priors = [20, 45, 90];   % degrees; smaller = stronger pull to horizontal

%% ---- Setup -----------------------------------------------------------------

phi_deg = linspace(-90, 270, 720);    % full circle, displayed as [-90, 270]
phi_rad = deg2rad(phi_deg);

% Helper: von Mises pdf  VM(phi ; mu, kappa)
%   exp(kappa * cos(phi - mu))  /  (2*pi * I0(kappa))
vm = @(phi_rad, mu_rad, kappa) ...
    exp(kappa .* cos(phi_rad - mu_rad)) ./ (2*pi * besseli(0, kappa));

% Simple sigma -> kappa  (kappa ~ 1/sigma_rad^2)
sig2k = @(sigma_deg) 1 ./ deg2rad(sigma_deg).^2;

%% ---- Figure layout ---------------------------------------------------------
% Left column  : Model 1 (no prior) for the three listeners
% Right column : Model 2 (with prior) for the typical listener, 3 prior strengths

n_rows = max(size(listeners, 1), numel(sigma_priors));

% Colours
c_correct   = [0.2  0.5  0.9];        % blue   - correct-hemifield component
c_confused  = [0.9  0.3  0.2];        % red    - front-back confusion component
c_total     = [0.1  0.9  0.1, 0.3];   % black  - total (with transparency)
c_target    = [0.1  0.1  0.1];        % green  - target location
c_resp       = [0.85 0.1  0.5];        % magenta - example response
% Figure 2 only colours
c_prior      = [0.9  0.6  0.1];       % amber  - prior
c_posterior  = [0.5  0.1  0.7];       % purple - posterior (= Model 2)

%% ---- Figure 1: Model 1 (no prior), one row per listener -------------------

figure('Name', 'Model 1 - No prior');
tiledlayout(size(listeners, 1), 1, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = gobjects(size(listeners, 1), 1);   % store axes for shared y-limits

for i = 1:size(listeners, 1)
    sigma = listeners{i, 1};
    w     = listeners{i, 2};
    label = listeners{i, 3};

    k = sig2k(sigma);

    mu1_rad = deg2rad(phi_t);          % correct hemifield
    mu2_rad = deg2rad(180 - phi_t);    % mirrored (front-back confusion)

    comp1 = w       * vm(phi_rad, mu1_rad, k);
    comp2 = (1 - w) * vm(phi_rad, mu2_rad, k);
    total = comp1 + comp2;

    ax1(i) = nexttile;
    hold on;
    plot(phi_deg, comp1, 'Color', c_correct,  'LineWidth', 1.0, 'DisplayName', 'w \cdot VM(\phi_t, \kappa)');
    plot(phi_deg, comp2, 'Color', c_confused, 'LineWidth', 1.0, 'DisplayName', '(1-w) \cdot VM(\pi-\phi_t, \kappa)');
    plot(phi_deg, total, 'Color', c_total,    'LineWidth', 2.0, 'DisplayName', 'p(\phi | \phi_t)');
    xline(phi_t, 'Color', c_target, 'LineWidth', 1.5, 'HandleVisibility', 'off');

    % --- Example response: show likelihood at phi_resp ---
    ll_resp = interp1(phi_deg, total, phi_resp);
    xline(phi_resp, '--', 'Color', c_resp, 'LineWidth', 1.2, 'HandleVisibility', 'off');
    plot(phi_resp, ll_resp, 'o', 'Color', c_resp, 'MarkerFaceColor', c_resp, ...
        'MarkerSize', 7, 'DisplayName', sprintf('\\phi_{resp} = %d° - probability (p)', phi_resp));
    text(phi_resp + 5, ll_resp, sprintf('  p = %.3f', ll_resp), ...
        'Color', 'black', 'FontSize', 8, 'VerticalAlignment', 'middle');

    xlim([-90 270]); xlabel('Polar angle (°)'); ylabel('p(\phi | \phi_t)');
    title(sprintf('Model 1  |  %s\n\\sigma = %d°,  w = %.2f', label, sigma, w));
    if i == 1
        legend('Location', 'northeast', 'FontSize', 8);
    end
end
linkaxes(ax1, 'y');

%% ---- Figure 2: Model 2 (with prior), one row per prior strength -----------

figure('Name', 'Model 2 - With horizontal prior');
tiledlayout(numel(sigma_priors), 1, 'TileSpacing', 'compact', 'Padding', 'compact');

sigma_listener = listeners{2, 1};
w_listener     = listeners{2, 2};
k_listener     = sig2k(sigma_listener);

ax2 = gobjects(numel(sigma_priors), 1);   % store axes for shared y-limits

for j = 1:numel(sigma_priors)
    sp = sigma_priors(j);
    k_prior = sig2k(sp);

    mu1_rad = deg2rad(phi_t);
    mu2_rad = deg2rad(180 - phi_t);

    % Prior: equal weight on front (0 deg) and back (180 deg)
    prior = 0.5 * vm(phi_rad, 0,  k_prior) ...
          + 0.5 * vm(phi_rad, pi, k_prior);

    % Unnormalised posterior
    comp1 = w_listener       * vm(phi_rad, mu1_rad, k_listener);
    comp2 = (1 - w_listener) * vm(phi_rad, mu2_rad, k_listener);
    likelihood = comp1 + comp2;

    posterior_unnorm = likelihood .* prior;

    % Normalise so area under curve = 1
    dphi = phi_rad(2) - phi_rad(1);
    posterior = posterior_unnorm ./ (sum(posterior_unnorm) * dphi);

    ax2(j) = nexttile;
    hold on;
    plot(phi_deg, likelihood, 'Color', c_total,     'LineWidth', 1.2, 'DisplayName', 'p(\phi | \phi_t)');
    plot(phi_deg, prior,      'Color', c_prior,      'LineWidth', 1.2, 'DisplayName', 'prior(\phi)');
    plot(phi_deg, posterior,  'Color', c_posterior,  'LineWidth', 2.0, 'DisplayName', 'p(\phi | \phi_t) \cdot prior(\phi)');
    xline(phi_t, 'Color', c_target, 'LineWidth', 1.5, 'HandleVisibility', 'off');

    % --- Example response: show posterior density at phi_resp ---
    post_resp = interp1(phi_deg, posterior, phi_resp);
    xline(phi_resp, '--', 'Color', c_resp, 'LineWidth', 1.2, 'HandleVisibility', 'off');
    plot(phi_resp, post_resp, 'o', 'Color', c_resp, 'MarkerFaceColor', c_resp, ...
        'MarkerSize', 7, 'DisplayName', sprintf('\\phi_{resp} = %d° - probability (p)', phi_resp));
    text(phi_resp + 5, post_resp, sprintf('  p = %.3f', post_resp), ...
        'Color', 'black', 'FontSize', 8, 'VerticalAlignment', 'middle');

    xlim([-90 270]); xlabel('Polar angle (°)'); ylabel('p(\phi | \phi_t) \cdot prior(\phi)');
    title(sprintf('Model 2  |  Typical listener\n\\sigma_{prior} = %d°  (prior strength)', sp));
    if j == 1
        legend('Location', 'northeast', 'FontSize', 8);
    end
end
linkaxes(ax2, 'y');

% Add titles to each figure
figure(1); sgtitle(sprintf('Model 1 — No prior  (target = %d°)', phi_t), 'FontSize', 13, 'FontWeight', 'bold');
figure(2); sgtitle(sprintf('Model 2 — With horizontal prior  (target = %d°)', phi_t), 'FontSize', 13, 'FontWeight', 'bold');


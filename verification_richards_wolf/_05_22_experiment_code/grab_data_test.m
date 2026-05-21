%% simulate_aberration_series.m
% Iterates over a list of aberrations and plots the simulated E field.
% No SLM, ScanImage, DLLs, or external hardware required.

%% ---------------- USER SETTINGS ----------------

outputFolder = fullfile(pwd, 'simulated_E_fields');
if ~exist(outputFolder, 'dir'); mkdir(outputFolder); end

% Simulated SLM dimensions
slmWidth  = 1024;
slmHeight = 1024;

saveFigures = true;

%% ---------------- DEFINE ABERRATIONS ----------------
% nm:     [n m] Zernike mode pairs corresponding to Z_n^m
% coeffs: strengths in waves, same order as nm

aberrations = struct([]);

aberrations(1).name   = 'wavefront_1';
aberrations(1).nm     = [
    2  2;  
    4  0  
];
aberrations(1).coeffs = [
    0.3;
    0.5
];

aberrations(2).name   = 'wavefront_2';
aberrations(2).nm     = [
    3 -1; 
    4  0 
];
aberrations(2).coeffs = [
    0.5;
    0.5
];

aberrations(3).name   = 'wavefront_3';
aberrations(3).nm     = [
    3 -1;  
    2 -2 
];
aberrations(3).coeffs = [
    0.5;
    0.3
];

aberrations(4).name   = 'wavefront_4';
aberrations(4).nm     = [
    3 -1;  
    2  2 
];
aberrations(4).coeffs = [
    0.5;
    0.3
];

aberrations(5).name   = 'wavefront_5';
aberrations(5).nm     = [
    3 -1;  
    2 -2;  
    4  0   
];
aberrations(5).coeffs = [
    0.5;
    0.3;
    0.5
];

aberrations(6).name   = 'wavefront_6';
aberrations(6).nm     = [
    3 -1;  
    2  2;  
    4  0   
];
aberrations(6).coeffs = [
    0.5;
    0.3;
    0.5
];


%% ---------------- DEFINE CIRCULAR PUPIL ----------------

center_x = 411;
center_y = 457;

waist_x  = 376;
waist_y  = 431;

radius = min(waist_x, waist_y)/2;


%% ---------------- SIMULATE E FIELDS ----------------

log = struct([]);

for k = 1:numel(aberrations)

    name   = aberrations(k).name;
    nm     = aberrations(k).nm;
    coeffs = aberrations(k).coeffs(:);

    fprintf('\n=== Aberration %d/%d: %s ===\n', ...
        k, numel(aberrations), name);

    if isempty(nm)
        % Flat phase
        E = ones(slmWidth * slmHeight, 1);
        phase_waves = zeros(slmWidth * slmHeight, 1);

        pupil = makeCircularPupil( ...
            slmWidth, slmHeight, [center_x, center_y], radius);

    else
        if size(nm, 1) ~= numel(coeffs)
            error('Aberration "%s" has %d modes but %d coefficients.', ...
                name, size(nm, 1), numel(coeffs));
        end

        [Z, pupil] = buildZernikeBasisOnSLM( ...
            slmWidth, slmHeight, nm, [center_x, center_y], radius);

        [E, phase_waves] = makeSLMField(Z, coeffs);
    end


    %% -------- Add simulated background grating outside the pupil only --------

    E = addBG(E, slmWidth, slmHeight, pupil);


    %% -------- Reshape for plotting --------

    E_img = reshape(E, slmHeight, slmWidth);
    phase_waves_img = reshape(phase_waves, slmHeight, slmWidth);

    phase_rad_img = angle(E_img);
    real_img = real(E_img);
    imag_img = imag(E_img);


    %% -------- Plot simulated E field --------

    fig = figure('Name', name, 'Color', 'w');
    tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    imagesc(phase_waves_img);
    axis image off;
    colorbar;
    title('Pupil phase [waves]');

    nexttile;
    imagesc(phase_rad_img);
    axis image off;
    colorbar;
    title('Total phase angle(E) [rad]');

    nexttile;
    imagesc(real_img);
    axis image off;
    colorbar;
    title('Re(E)');

    nexttile;
    imagesc(imag_img);
    axis image off;
    colorbar;
    title('Im(E)');

    sgtitle(sprintf('%s: simulated E field', name), 'Interpreter', 'none');


    %% -------- Save simulated data / figure --------

    if saveFigures
        saveas(fig, fullfile(outputFolder, sprintf('%02d_%s_Efield.png', k, name)));
    end

    save(fullfile(outputFolder, sprintf('%02d_%s_Efield.mat', k, name)), ...
        'E', 'phase_waves', 'pupil', 'nm', 'coeffs');


    %% -------- Log --------

    log(k).name = name;
    log(k).nm = nm;
    log(k).coeffs = coeffs;
    log(k).phase_waves = phase_waves;
    log(k).pupil = pupil;

    fprintf('Plotted %s\n', name);
end

save(fullfile(outputFolder, 'aberration_series_log.mat'), ...
    'aberrations', 'log');

disp('Done.');


%% ================= LOCAL FUNCTIONS =================

function [E, phase_waves] = makeSLMField(Zbasis, coeffs)
% Zbasis is [Npix x nModes].
% coeffs are in waves.
%
% phase_waves is dimensionless phase in cycles/waves.
% E is the complex field exp(i 2pi phase_waves).

    phase_waves = Zbasis * coeffs;
    E = exp(1i * 2*pi * phase_waves);
end


function E_with_bg = addBG(E, width, height, pupil)
% Hardware-free replacement for the old addBG.
% Adds a simple phase grating outside the pupil only.

    BP_bg_grating = 8;

    [X, ~] = meshgrid(1:width, 1:height);

    % Simulated phase grating in waves.
    % Period = BP_bg_grating pixels.
    bg_phase_waves = mod(X, BP_bg_grating) / BP_bg_grating;

    pupilVec = pupil(:);
    bg_phase_waves_vec = bg_phase_waves(:);

    E_bg = ones(width * height, 1);
    E_bg(~pupilVec) = exp(1i * 2*pi * bg_phase_waves_vec(~pupilVec));

    E_with_bg = E .* E_bg;
end


function [Zbasis, pupil] = buildZernikeBasisOnSLM(width, height, nmPairs, center, Rpix)
% Returns Zbasis [Npix x nModes] in WAVES.
%
% nmPairs should be [n m], corresponding to Z_n^m.
%
% The circular pupil is defined by:
%   center = [center_x, center_y]
%   radius = Rpix
%
% Inside the pupil:
%   rho = r / Rpix
%
% Outside the pupil:
%   Z = 0

    nModes = size(nmPairs, 1);
    Npix = width * height;

    [X, Y] = meshgrid(1:width, 1:height);

    x = X - center(1);
    y = Y - center(2);

    rho = sqrt(x.^2 + y.^2) / Rpix;
    ang = atan2(y, x);

    pupil = rho <= 1;

    Zbasis = zeros(Npix, nModes);

    for k = 1:nModes
        n = nmPairs(k,1);
        m = nmPairs(k,2);

        Z = zeros(height, width);

        Z(pupil) = zernike_nm(n, m, rho(pupil), ang(pupil));

        Zbasis(:,k) = Z(:);
    end
end


function pupil = makeCircularPupil(width, height, center, radius)
% Returns logical circular pupil mask.

    [X, Y] = meshgrid(1:width, 1:height);

    x = X - center(1);
    y = Y - center(2);

    pupil = sqrt(x.^2 + y.^2) <= radius;
end


function Z = zernike_nm(n, m, rho, theta)
% Real Zernike Z_n^m on unit disk.
%
% m > 0: R_n^{|m|}(rho) cos(|m| theta)
% m < 0: R_n^{|m|}(rho) sin(|m| theta)
% m = 0: R_n^0(rho)

    mabs = abs(m);

    if n < 0 || mabs > n || mod(n - mabs, 2) ~= 0
        Z = zeros(size(rho));
        return;
    end

    R = zernike_radial(n, mabs, rho);

    if m == 0
        Z = R;
    elseif m > 0
        Z = R .* cos(mabs * theta);
    else
        Z = R .* sin(mabs * theta);
    end
end


function R = zernike_radial(n, m, rho)
% Radial polynomial R_n^m(rho)

    if mod(n-m,2) ~= 0
        R = zeros(size(rho));
        return;
    end

    R = zeros(size(rho));

    smax = (n - m) / 2;

    for s = 0:smax
        c = (-1)^s * factorial(n - s) / ...
            ( factorial(s) * ...
              factorial((n + m)/2 - s) * ...
              factorial((n - m)/2 - s) );

        R = R + c * rho.^(n - 2*s);
    end
end
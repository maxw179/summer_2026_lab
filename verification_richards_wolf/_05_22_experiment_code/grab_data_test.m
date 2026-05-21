%% test_aberration_series_offline.m
% Hardware-free version of acquire_aberration_series.m.
%
% Builds Zernike phase maps and complex E fields without connecting to
% the microscope, SLM, ScanImage, Blink SDK, or any external hardware.
%
% Outputs:
%   log(k).phase_waves       [1024 x 1024], phase in waves
%   log(k).phase_rad         [1024 x 1024], phase in radians
%   log(k).E                 [1024 x 1024], complex field exp(i phase)
%   log(k).E_with_bg         [1024 x 1024], complex field with background grating
%   log(k).pupil             [1024 x 1024], logical circular pupil

clear; clc; close all;

%% ---------------- USER SETTINGS ----------------



N = 1024;

% Whether to add a simple software-generated grating outside the pupil.
addBackgroundGrating = true;

% Background grating settings.
% This replaces the hardware-dependent generateGratings(...) call.
bgPeriodPixels = 8;       % grating period in pixels
bgDirection    = 'x';     % 'x' or 'y'

% Display settings
makePlots = true;
plotAberrationIndex = 1;  % choose which aberration to visualize


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


%% ---------------- BUILD TEST FIELDS ----------------

log = struct([]);

for k = 1:numel(aberrations);

    name   = aberrations(k).name;
    nm     = aberrations(k).nm;
    coeffs = aberrations(k).coeffs(:);

    fprintf('\n=== Aberration %d/%d: %s ===\n', ...
        k, numel(aberrations), name);

    if isempty(nm)
        pupil = makeCircularPupil(N, N, [center_x, center_y], radius);

        phase_waves_vec = zeros(N*N, 1);
        E_vec = ones(N*N, 1);

    else
        if size(nm, 1) ~= numel(coeffs)
            error('Aberration "%s" has %d modes but %d coefficients.', ...
                name, size(nm, 1), numel(coeffs));
        end

        [Z, pupil] = buildZernikeBasisOnSLM( ...
            N, N, nm, ...
            'Center', [center_x, center_y], ...
            'Radius', radius);

        [E_vec, phase_waves_vec] = makeSLMField(Z, coeffs);
    end

    phase_waves = reshape(phase_waves_vec, N, N);
    phase_rad   = 2*pi*phase_waves;
    E           = reshape(E_vec, N, N);

    if addBackgroundGrating
        bg_phase_rad = makeBackgroundGratingPhase( ...
            N, N, bgPeriodPixels, bgDirection);

        E_bg = exp(1i * bg_phase_rad);

        % Apply background grating outside the circular pupil only.
        E_with_bg = E;
        E_with_bg(~pupil) = E_bg(~pupil);
    else
        bg_phase_rad = zeros(N, N);
        E_with_bg = E;
    end

    log(k).name = name;
    log(k).nm = nm;
    log(k).coeffs = coeffs;

    log(k).phase_waves = phase_waves;
    log(k).phase_rad = phase_rad;
    log(k).E = E;
    log(k).E_with_bg = E_with_bg;
    log(k).pupil = pupil;
    log(k).bg_phase_rad = bg_phase_rad;

    fprintf('Built %s\n', name);
end


%% ---------------- OPTIONAL PLOTS ----------------

if makePlots
    k = plotAberrationIndex;

    phase_waves = log(k).phase_waves;
    phase_rad   = log(k).phase_rad;
    E           = log(k).E;
    E_with_bg   = log(k).E_with_bg;
    pupil       = log(k).pupil;

    figure;
    imagesc(phase_waves);
    axis image;
    colorbar;
    title(sprintf('%s: phase in waves', log(k).name));
    xlabel('x pixel');
    ylabel('y pixel');

    figure;
    imagesc(phase_rad);
    axis image;
    colorbar;
    title(sprintf('%s: phase in radians', log(k).name));
    xlabel('x pixel');
    ylabel('y pixel');

    figure;
    imagesc(angle(E));
    axis image;
    colorbar;
    title(sprintf('%s: angle(E), no background grating', log(k).name));
    xlabel('x pixel');
    ylabel('y pixel');

    figure;
    imagesc(angle(E_with_bg));
    axis image;
    colorbar;
    title(sprintf('%s: angle(E with background grating)', log(k).name));
    xlabel('x pixel');
    ylabel('y pixel');

    figure;
    imagesc(pupil);
    axis image;
    colorbar;
    title('Circular pupil mask');
    xlabel('x pixel');
    ylabel('y pixel');
end

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


function [Zbasis, pupil] = buildZernikeBasisOnSLM(width, height, nmPairs, varargin)
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
%
% No RMS normalization is applied.

    p = inputParser;
    addParameter(p, 'Center', [width/2, height/2]);
    addParameter(p, 'Radius', min(width,height)/2);
    parse(p, varargin{:});

    center = p.Results.Center;
    Rpix   = p.Results.Radius;

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


function bg_phase_rad = makeBackgroundGratingPhase(width, height, periodPixels, direction)
% Software-only replacement for the external generateGratings call.
%
% Returns a phase ramp in radians.
%
% periodPixels controls how many pixels correspond to one 2pi phase cycle.

    [X, Y] = meshgrid(1:width, 1:height);

    switch lower(direction)
        case 'x'
            coord = X;
        case 'y'
            coord = Y;
        otherwise
            error('direction must be either "x" or "y".');
    end

    bg_phase_rad = 2*pi * coord / periodPixels;
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
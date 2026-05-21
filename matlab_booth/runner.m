%% runner.m
% Runs the Booth modal AO algorithm

%% ---------------- USER SETTINGS ----------------

outputFolder = 'C:\Users\ScanImage\Documents\Diego_Max\wolf_verify_z0\';
if ~exist(outputFolder, 'dir'); mkdir(outputFolder); end

pauseAfterSLM = 0.2;     % seconds after writing SLM before grabbing frame
num_frame  = 1;       % set >1 if you want frame averaging

siHeight = 512;
siWidth  = 512;

lut1024 = 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm6748_at1300_1stOrder_031325_3.lut';
lut512  = 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\512x512_linearVoltage.LUT';


%% ---------------- BOOTH PARAMETERS ----------------

numIterations = 10;       % outer iterations
bias          = 0.20;    % initial bias (waves)
biasMin       = 0.025;    % min bias (waves)
biasDecay     = 4/5;     % bias *= biasDecay each iteration

%zeta     = 0.1;         % step gain
%stepClip = 0.5;          % max abs step (waves)

%% ---------------- MODE LIST (n,m) ----------------
% Choose modes you want to correct. 

% Modes allowed in the PRESET aberration (can be higher order / arbitrary)
% Example: include up to n=6 (you can add/remove freely)
nmPreset = [
    2 -2; 2 0; 2 2;
    3 -3; 3 -1; 3 1; 3 3;
    4 -4; 4 -2; 4 0; 4 2; 4 4;
    5 -5; 5 -3; 5 -1; 5 1; 5 3; 5 5;
    6 -6; 6 -4; 6 -2; 6 0; 6 2; 6 4; 6 6
];

nmCorr = [4 0;
    2 0;];



%% ---------------- INIT SLM ----------------

slm = SLMController( ...
    'BoardNumber', 1, ...
    'BitDepth', 12, ...
    'InvPiOver128', 128/pi, ...
    'LUT1024', lut1024, ...
    'LUT512',  lut512);

slm.load();

% Mask: match your aperture choice
% Option A: full mask
% slm.setMask(uint8(ones(slm.Width*slm.Height,1)));

% Option B: circular mask (edit radius as needed)

%% ---------------- DEFINE CIRCULAR PUPIL ----------------

center_x = 411;
center_y = 457;

waist_x  = 376;
waist_y  = 431;

radius = min(waist_x, waist_y)/2;

% Keep this if SLMController uses its internal mask during writeEfield.
slm.setCircularMask('Center', [center_x center_y], 'Radius', radius);


%% ---------------- INIT SCANIMAGE ----------------

si = ScanImageController(hSI, ...
    'PollPeriodSec', 2, ...
    'FallbackSize', [siHeight siWidth]);

%% ---------------- BUILD ZERNIKE BASES ON SLM ----------------
% Zcorr:   basis Booth will optimize over
% Zpreset: basis used to construct the injected preset aberration

[Zcorr, pupil] = buildZernikeBasisOnSLM( ...
    slm.Width, slm.Height, nmCorr, [center_x, center_y], radius);

[Zpreset, pupil] = buildZernikeBasisOnSLM( ...
    slm.Width, slm.Height, nmPreset, [center_x, center_y], radius);

nCorr   = size(Zcorr, 2);
nPreset = size(Zpreset, 2);

% ---------------- PRESET ABERRATION MODE ----------------
usePresetAberration = true;

rng(0);
%aPreset = 0.1 * randn(nPreset, 1)   % preset coeffs (waves) on nmPreset basis
aPreset = zeros(25, 1);
% aPreset(5) = 0.2;
% aPreset(6) = 0.2;
%aPreset(10) = 0.5;
%aPreset(1) = 0.5;
%aPreset(5) = 0.5;
% aPreset(11) = 0.2;


c = zeros(nCorr, 1);                  % correction coeffs (waves) on nmCorr basis

% Log arrays
c_log = zeros(numIterations+1, nCorr);
M0_log  = zeros(numIterations + 2, 1);
bias_log = zeros(numIterations,1);
c_log(1,:) = c;

img0 = si.grabFrame();
saveTiff16(img0, fullfile(outputFolder, 'Img_Booth_ground_truth.tif'));
M_ground = imageMetric(img0);
M0_log(1) = M_ground;

disp('===== Booth correction with preset SLM aberration =====');
disp(['Modes: ', num2str(nCorr), ' | Iterations: ', num2str(numIterations)]);

% ---------------- MAIN BOOTH LOOP ----------------
for it = 1:numIterations
    fprintf('\n=== Iteration %d/%d | bias = %.3f waves ===\n', it, numIterations, bias);

   
    bias_log(it) = bias;

    % ---- baseline at current c ----
    E0 = makeSLMField(Zcorr, c, usePresetAberration, Zpreset, aPreset);
    E0 = addBG(E0, slm, pupil);
    slm.writeEfield(E0);
    pause(pauseAfterSLM);

    img0 = 0;
    for frame = 1:num_frame
        img0 = img0 + si.grabFrame();
    end
    img0 = img0/num_frame;

    saveTiff16(img0, fullfile(outputFolder, ['Img_Booth_it' num2str(it) '.tif']));
    M0 = imageMetric(img0);
    M0_log(it + 1) = M0;
    fprintf('Baseline metric M0 = %.6g\n', M0);

    steps = zeros(1, nCorr);

    % ---- optimize each mode (2N+1) ----
    for m = 1:nCorr
        % +bias
        c_plus = c; c_plus(m) = c_plus(m) + bias;
        Eplus = makeSLMField(Zcorr, c_plus, usePresetAberration, Zpreset, aPreset);
        Eplus = addBG(Eplus, slm, pupil);
        slm.writeEfield(Eplus);
        pause(pauseAfterSLM);

        img0 = 0;
        for frame = 1:num_frame
            img0 = img0 + si.grabFrame();
        end
        img0 = img0/num_frame;
        Mplus = imageMetric(img0);

        % -bias
        c_minus = c; c_minus(m) = c_minus(m) - bias;
        Eminus = makeSLMField(Zcorr, c_minus, usePresetAberration, Zpreset, aPreset);
        Eminus = addBG(Eminus, slm, pupil);
        slm.writeEfield(Eminus);
        pause(pauseAfterSLM);

        img0 = 0;
        for frame = 1:num_frame
            img0 = img0 + si.grabFrame();
        end
        img0 = img0/num_frame;
        Mminus = imageMetric(img0);

        % fit quadratic to metric vs coefficient offset
        xfit = [-bias, 0, bias];
        yfit = [Mminus, M0, Mplus];

        %p = polyfit(xfit, yfit, 2);   % p(1)x^2 + p(2)x + p(3)
        p = fit(xfit', yfit', 'poly2', 'Lower', [-Inf, -Inf, -Inf], 'Upper', [0, Inf, Inf]); %from 

        %vertex of parabola
        if abs(p.p1) > eps
            step = -p.p2 / (2*p.p1);
        else
            step = 0;
        end

        %limit large jumps
        step = max(min(step, bias), -bias);
        steps(m) = step;

        fprintf('  mode %2d: M+ %.4g | M- %.4g | step %+0.4f -> c=%.4f\n', ...
            m, Mplus, Mminus, step, c(m) + step);

    end
    c = c + steps';
    c_log(it+1,:) = c;

    % decay bias
    bias = max(biasMin, bias * biasDecay);
end

%% ---------------- FINAL WRITE + SAVE ----------------

EF = makeSLMField(Zcorr, c, usePresetAberration, Zpreset, aPreset);
EF = addBG(EF, slm, pupil);
slm.writeEfield(EF);
pause(pauseAfterSLM);
imgF = si.grabFrame();
MF = imageMetric(imgF);
M0_log(numIterations + 2) = MF;

fprintf('\nFinal metric MF = %.6g\n', MF);

%Compare to ideal cancellation (since preset is expressed in same basis)
if usePresetAberration
    phiResidual = Zcorr*c + Zpreset*aPreset;
    phi_Residual = phiResidual(pupil(:));  % residual phase in waves on SLM grid
    fprintf('Residual phase RMS on pupil (waves): %.5f\n', rms(phiResidual));
end


saveTiff16(imgF, fullfile(outputFolder, 'Img_BoothPreset_Final.tif'));

save(fullfile(outputFolder, 'booth_preset_log.mat'), ...
   'c','aPreset','c_log','M0_log','bias_log','MF');

%Cleanup
delete(slm);

disp('Done.');

%% ================= LOCAL FUNCTIONS =================

function E = makeSLMField(Zcorr, cCorr, usePreset, Zpreset, aPreset)
% phase_waves = Zcorr*cCorr + Zpreset*aPreset (if enabled)
    phase_waves = Zcorr * cCorr;
    if usePreset
        phase_waves = phase_waves + (Zpreset * aPreset);
    end
    E = exp(1i * 2*pi * phase_waves);
end

function M = imageMetric(img)
    % Mean squared img intensity
    img = double(img);
    p = prctile(img, 99);
    mask = img > p;
    M = mean(mean(img(mask)));
end

function [E_with_bg] = addBG(E, slm, pupil)
    board_number = slm.BoardNumber;
    depth = calllib('Blink_C_wrapper', 'Get_image_depth', board_number); 
    Bytes = depth/8;

    WFC = libpointer('uint8Ptr', zeros(slm.Width * slm.Height * Bytes, 1));

    BP_bg_grating = 8;
    bg_grating = generateGratings(BP_bg_grating, 3, false, WFC);

    pupilVec = pupil(:);
    bgVec = bg_grating(1).value(:);

    E_bg_slm = zeros(slm.Width * slm.Height, 1, 'uint8');

    % Apply grating only outside the circular pupil.
    E_bg_slm(~pupilVec) = bgVec(~pupilVec);

    E_bg = exp(1j * double(E_bg_slm) * pi/128);

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
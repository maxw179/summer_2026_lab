%% grab_data.m
% Iterates over a list of aberrations and takes one image under each one.
% Each aberration can contain multiple Zernike mode/strength pairs.

%% ---------------- USER SETTINGS ----------------

outputFolder = 'C:\Users\ScanImage\Documents\Diego_Max\wolf_verify_z0\';
if ~exist(outputFolder, 'dir'); mkdir(outputFolder); end

pauseAfterSLM = 0.2;     % seconds after writing SLM before grabbing frame
numFramesAvg  = 1;       % set >1 if you want frame averaging

siHeight = 512;
siWidth  = 512;

lut1024 = 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm6748_at1300_1stOrder_031325_3.lut';
lut512  = 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\512x512_linearVoltage.LUT';


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


%% ---------------- INIT SLM ----------------

slm = SLMController( ...
    'BoardNumber', 1, ...
    'BitDepth', 12, ...
    'InvPiOver128', 128/pi, ...
    'LUT1024', lut1024, ...
    'LUT512',  lut512);

slm.load();


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


%% ---------------- ACQUIRE IMAGES ----------------

log = struct([]);

try
    for k = 1:numel(aberrations)

        name   = aberrations(k).name;
        nm     = aberrations(k).nm;
        coeffs = aberrations(k).coeffs(:);

        fprintf('\n=== Aberration %d/%d: %s ===\n', ...
            k, numel(aberrations), name);

        if isempty(nm)
            % Flat phase
            E = ones(slm.Width * slm.Height, 1);
            phase_waves = zeros(slm.Width * slm.Height, 1);

            pupil = makeCircularPupil( ...
                slm.Width, slm.Height, [center_x, center_y], radius);

        else
            if size(nm, 1) ~= numel(coeffs)
                error('Aberration "%s" has %d modes but %d coefficients.', ...
                    name, size(nm, 1), numel(coeffs));
            end

            [Z, pupil] = buildZernikeBasisOnSLM( ...
                slm.Width, slm.Height, nm, [center_x, center_y], radius);

            [E, phase_waves] = makeSLMField(Z, coeffs);
        end


        %% -------- Add background grating outside the pupil only --------

        E = addBG(E, slm, pupil);

        %% -------- Write aberration to SLM --------

        slm.writeEfield(E);
        pause(pauseAfterSLM);


        %% -------- Grab image, optionally averaging multiple frames --------

        img = 0;

        for frame = 1:numFramesAvg
            img = img + double(si.grabFrame());
        end

        img = img / numFramesAvg;


        %% -------- Save image --------

        filename = sprintf('Img_%02d_%s.tif', k, name);
        saveTiff16(img, fullfile(outputFolder, filename));


        %% -------- Log --------

        log(k).name = name;
        log(k).nm = nm;
        log(k).coeffs = coeffs;
        log(k).filename = filename;
        log(k).phase_waves = phase_waves;
        log(k).pupil = pupil;

        fprintf('Saved %s\n', filename);
    end

    save(fullfile(outputFolder, 'aberration_series_log.mat'), ...
        'aberrations', 'log');

catch ME
    delete(slm);
    rethrow(ME);
end

delete(slm);
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
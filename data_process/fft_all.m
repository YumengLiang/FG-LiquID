%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DESCRIPTION:
% Data extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Author: Yumeng Liang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mxTargets,m1, mxMag] = fft_all ( load_path, name)
    load(load_path);
    BW = 6e9; %bandwidth
    Rres = 3.0e8/(2*BW); %range resolution
    samples = 256; 
    Zeropad = samples; 
    Rmax = Rres * samples; 
    range = linspace(0,1-1/Zeropad,Zeropad)*Rmax;
    Hann_window = hann(samples,'periodic');
    ScaleHannWin = 1/sum(Hann_window);
    str1 = strcat('./feature/',name);

    Nbut = 4;                                                                        % 4th order of filter
    Wn = .05;                                                                        % cutoff frequency
    [b,a] = butter(Nbut,Wn,'high');                                                  % butter(order of filter, cutoff frequency, type of filter);
    Rx1 = filter(b,a,Rx1);                                                     % filtering
    Rx2 = filter(b,a,Rx2);
    Rx3 = filter(b,a,Rx3);
    Rx4 = filter(b,a,Rx4);

    %% apply fft 
    Rx4 = fft(Rx4.*Hann_window',Zeropad);
    Rx4 = Rx4(1:length(range))*1*ScaleHannWin;
    Rx3 = fft(Rx3.*Hann_window',Zeropad);
    Rx3 = Rx3(1:length(range))*1*ScaleHannWin;
    Rx2 = fft(Rx2.*Hann_window',Zeropad);
    Rx2 = Rx2(1:length(range))*1*ScaleHannWin;
    Rx1 = fft(Rx1.*Hann_window',Zeropad);
    Rx1 = Rx1(1:length(range))*1*ScaleHannWin;
    mxFFT = [transpose(Rx3), transpose(Rx4) , transpose(Rx2) , transpose(Rx1)];

    %% get maximum RSS from the desired range 
    mxMag = abs(mxFFT);
    [m1, p1] = max(mxMag(12:20,:));
        
    %% get the feature as input of CNN
    num =4; % the number of range bins around the one with maximum RSS 

    save_mat = zeros(1+num*2,4,4); 
    % save_mat is the output features
    % 1+num*2: the number of output range bins
    % 4 x 4: the 4 features(distance, azimuth angle, elevation angle, RSS) of 4 antennas 
    rvLinInd = zeros(2,size(mxFFT,2));

    for i=1:4
        p1(i) = p1(i)+11;
    end
    
    for i = -num:num
        rvPeakInd = p1+i;
        rvLinInd(1,:) = sub2ind(size(mxFFT), rvPeakInd, 1:size(mxFFT,2));     % linear index to pick out peaks from matrix
        rvLinInd(2,:) = sub2ind(size(mxFFT), rvPeakInd, [2:size(mxFFT,2),1]); % linear index to pick out peaks from matrix
        rvPh = unwrap(angle(mxFFT(rvLinInd)),[],1);   % get phase information only
        rvPhDelt = diff(-rvPh,1,1);                   % get phase difference 
        
        cvRange = range(rvPeakInd);
        scC0 = 3e8;             % speed of light in m/s
        scSpace = 3.5e-3;       % spacing between two receivers in m (scWL*.7 for scWL=5e-3, F=6GHz)
        scF1 = 64e9;
        scWL = scC0/scF1;       % maximum signal wavelength in m
        rvAngle = getAoA(rvPhDelt, scSpace/scWL);
        
        rvAngle([1 4]) = -rvAngle([1 4]);
        mxTargets = AoA(cvRange, rvAngle);
        
        mxMag_peak = ones(1,4);
        mxMag_peak(:,1) = mxMag(rvPeakInd(1),1);
        mxMag_peak(:,2) = mxMag(rvPeakInd(2),2);
        mxMag_peak(:,3) = mxMag(rvPeakInd(3),3);
        mxMag_peak(:,4) = mxMag(rvPeakInd(4),4);
        
        mxTargets = cat(1,mxTargets,mxMag_peak);
        if i==0
            mxTargets(:,:)
        end
        save_mat(i+num+1,:,:) = mxTargets;
    end
    save(str1,'save_mat');
end

function cvAlph = getAoA(cvPhDelt, scXi)
% cvPhDelt: phase deltas in rad
% scXi: normalized space between receivers, i.e. = S/WL with S: space in m and WL: wavelength in m
% cvAlph: AoA in rad, when looking from the receiver to the source:
% 0: broadside, -pi/2: left, pi/2: right

cvAlphSin = (cvPhDelt/(2*pi*scXi)); % normalize to xi (scSpace/scWL)
cvAlph = asin(cvAlphSin);
end

function [mxTargets] = AoA(rvR, rvAlph)
% Build (distance, azimuth angle, elevation angle) coordinates for each Rx
mxTargets = zeros(3,size(rvR,2));
mxTargets(:,1) = [rvR(1), rvAlph(1), rvAlph(4)];
mxTargets(:,2) = [rvR(2), rvAlph(1), rvAlph(2)];
mxTargets(:,3) = [rvR(3), rvAlph(3), rvAlph(2)];
mxTargets(:,4) = [rvR(4), rvAlph(3), rvAlph(4)];
end
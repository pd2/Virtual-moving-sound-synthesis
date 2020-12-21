% This script synthesizes stereo audio signals that simulate moving sound 
%  sources for a specified angular speed and path at a given resolution.
%  It first extracts the HRIR from the Wisconsin HRTF database, computes
%  the equalization required for diffuse field response, interpolates
%  linearly in frequency domain for the positions requested, resamples to
%  the output sampling rate required, convolves with input of user choice
%  and applies equalization together with global scale factor and then
%  concatenates the responses to stimulate movement in the virtual
%  soundscape and finally saves the output as wav file
%
% Inputs:
% 	db_path     - specify the path to the datasets
% 	inp_path    - specify the path to the input file
% 	out_path    - specify the path to store the output files
% 	subj_no     - specify wisconsin subject number from 1 to 5, 0 for avg
%   speed       - specify the speed at which to move the sound source
%   angular_res - specify the angular resolution for the movement
%   trajectory_list - specify the auditory object trajectory
% 	inp_file    - specify the file to obtain the response
%   op_filename_prefix - specify the prefix for the output file names
%   tot_duration - specify the total duration of the movement
%   specify positions at which to compute the response (in degrees)
%       azi_L_extreme_pos   - specify the extreme azimuth position in Left hemifield
%       azi_L_extreme_pos   - specify the extreme azimuth position in Left hemifield
%       elevation   - specify the fixed elevation at the sound moves
%
% Author:- Pradeep Dheerendra, Auditory Cognition Group,
% Institute of Neuroscience, Newcastle University, NCL, UK
% (C) copyright reserved 2015


% Version History:-
%  ver - DD MMM YYYY - Feature added
%  ---------------------------------
%  1.0 - 26 Jan 2015 - first implementation
%  2.0 - 27 Jan 2015 - bug fix for ILD in scalefactor
%  2.1 - 27 Jan 2015 - bug fix for ILD in global scalefactor
%  3.0 - 04 Feb 2015 - Frequency domain interpolation
%  4.0 - 09 Feb 2015 - concatenation of responses at different positions
%  5.0 - 11 Feb 2015 - support constant duration over constant distance
%  5.1 - 12 Feb 2015 - configurable start & stop positions for trajectories
%  6.0 - 13 Feb 2015 - diffuse-field response computation and equalization
%  6.2 - 24 Feb 2015 - optimized convolution by overlap-save method
%  6.3 - 25 Feb 2015 - bug fix on OSA, clean up, aid automation 
%  6.4 - 17 Mar 2015 - changed to overlap-add method in chunk processing
%  6.5 - 18 Mar 2015 - changed to block processing instead of chunks
%  6.6 - 19 Mar 2015 - cross fade during filter update to remove artifacts
%  6.7 - 22 Mar 2015 - optimized convolution loop within block processing
%  6.8 - 25 Mar 2015 - changes for malloc optimization, o/p filename convention
%  6.9 - 27 Mar 2015 - optimized o/p loop with precomputed dfr equalized HRTFs
%  7.0 - 27 Mar 2015 - verbosity levels added
%  7.1 - 30 Mar 2015 - sigmoid transition implemented
%  7.2 - 31 Mar 2015 - sigmoid transition cleanup, fixes in chunk size
%  7.3 - 01 Apr 2015 - generalized the overlap add to handle miniscule segments
%  7.4 - 07 Apr 2015 - support for resampling the HRTF database, error checks
%  7.5 - 10 Apr 2015 - support for spectrotemporal control output
%  7.6 - 28 Apr 2015 - code clean up
%  7.7 - 20 May 2015 - support for time domain interpolation
%  7.8 - 21 May 2015 - support for averaged HRTF


%% Configuration of this script - conditional execution to allow automation
if ~(exist('enable_automation','var') && enable_automation)
    
    % User defined configurations - used if script automation is disabled

    clear; close all; clc
    
    % specify the path to the datasets
    db_path = 'dataset/'; 
    
    % specify the path to the input file
    inp_path = 'input/';

    % specify the path to the output
    out_path = 'output/';
    
    % specify subject number from 1 to 5; 0- average of subjects
    subj_no = 1;
    
    % specify the speed in degrees per second
    speed = 100; % deg/s
    
    % specify the angular resolution at which to move the sound source
    angular_res = 1;  % in degrees
    
    % specify the auditory object trajectory
    % note: make a closed loop to avoid sudden jumps
    %   1: Left   to Center
    %   2: Center to Right
    %   3: Right  to Center
    %   4: Center to Left
    trajectory_list = [ 1 2 3 4];
    
    % specify the file to obtain the response
    % inp_file = 'modulated_am_noise_80_fs48k.wav';
      inp_file = 'Vintro.wav'; 
    
    % specify the prefix for the output file names
    op_filename_prefix = 'moving_sound_subj_';
    % op_filename_prefix = 'AM100hz_noise_';
    
end


%% default configurations

% specify the total duration in seconds
tot_duration = 4.9; % sec

% set the extreme positions for right and left hemifields
%   0 is straight ahead, +ve azimuths to the right while -ve to the left
azi_L_extreme_pos = -80;
azi_R_extreme_pos =  80;

% specify the elevation
%   0 is horizontal plane at ear level, +ve : above, -ve : below this plane
elevation = 0; % in degrees

% set the domain of interpolation
domain = 'freq'; % 'freq' 'time'

% set the method of interpolation
method = 'linear';

% set freq range for diffuse-field equalization
f_min = 300; % hz - lower limit
f_max = 8000;% hz - upper limit

% set the duration of each block for block processing of output
blk_dur = 10e-3; % in sec

% set flag to enable cross fade upon filter update
flag_enable_xfade = 0;

% set flag to enable sigmoid transition at lower resolutions
flag_sigmoid_trans = 0;

% set the duration of block for processing of output during sigmoid transition
blk_dur_sgmd = 1e-3; % in sec

% set   2 for regular output; 1 for spectrotemporal control
n_ch_out = 2;

% specify the file extension for the output files
file_extn = '.wav';

% set flag to enable debug code
flag_debug = 0;

% specify the path to output debug / tmp data
tmp_path = 'tmp/';

% flag to specify the amount of info printed on screen. Error msgs are not
% affected by this setting. % 0: silent  1: minimal  2: all  3: debug info
verbose = 3;



%% DO NOT MODIFY PARAMETERS/SCRIPT BEYOND THIS POINT.

%% database parameters

% samping frequency 
fs_db = 50e3;  % in Hz

n_ch = 2;   % stereo
n_bits = 16;% bitwidth

% list of subjects
subj_list = {'afw','sjx','sos','sou','sow'};
num_subjs = length(subj_list);

% angular parameters corresponding to this database in degrees

% elevation angles
first_ele = -50;
last_ele = 80;
jump_ele = 10;  % resolution of elevation

% azimuth angles 
first_azi = -170;
last_azi = 180;
jump_azi = 10;  % resolution of azimuth

% list of elevations and azimuth angles
ele_list = first_ele:jump_ele:last_ele;
azi_list = first_azi:jump_azi:last_azi;
tot_ele = length(ele_list);
tot_azi = length(azi_list);

% zenith - 90 degrees (directly overhead) is not listed in the above array
offset = 1;

% compute the total number of positions in the dataset
n_hrir = tot_azi * tot_ele + offset;

% length of the impulse response
% len_hn = 256;

%% initializations

if (subj_no > 5)
    disp('Error subj_no: out of range. select 1-5');
    return;
end

%  out_path = [out_path '/'];

% % anonmyous functions

% convert azimuth and elevation into database index
get_db_indx = @(az,el) ((190-az)/10-1) * tot_ele + (90-el)/10 + offset;

% convert angle to maintain conformance with dataset
conv_to_db_angle = @(inp_ang) (mod(inp_ang + 179, 360) - 179);

if n_ch_out > n_ch
    n_ch_out = n_ch;
end

if flag_debug
    verbose = 3;
end

if verbose >= 1
disp('Computing parameters');
end

%% Loading of data : database HRIR & Input signal
if verbose >= 2
disp('HRTF database');
end

% HRIR loading
if (subj_no == 0)
    % compute the mean of the HRTFs to obtain dataset average
    
    subj_name = 'avg';
    subj_nums = 1:num_subjs;
    
    % iterate over all subjs
    for s_no = subj_nums
        
        if (exist(db_path,'dir'))
            file = [db_path '/' subj_list{s_no} '.mat'];
            if (exist(file,'file'))
                load(file);
            else
                disp('Error: HRTF DB file does not exist');
                return;
            end
        else
            disp('Error: Input path does not exist');
            return;
        end
        
        % accumulate into temp var
        if s_no==1
            hrtf_l_sum = hrtf_l;
            hrtf_r_sum = hrtf_r;
        else
            hrtf_l_sum = hrtf_l_sum + hrtf_l;
            hrtf_r_sum = hrtf_r_sum + hrtf_r;
        end
    end
    
    % compute the mean
    hrtf_l = hrtf_l_sum/num_subjs;
    hrtf_r = hrtf_r_sum/num_subjs;
    
    % clear tmp vars
    clear hrtf_l_sum hrtf_r_sum;
else
    % load subject specific HRTF data
    subj_name = subj_list{subj_no};
    if (exist(db_path,'dir'))
        file = [db_path '/' subj_name '.mat'];
        if (exist(file,'file'))
            load(file);
        else
            disp('Error: HRTF DB file does not exist');
            return;
        end
    else
        disp('Error: Input path does not exist');
        return;
    end
end

% loading input signal
if (exist(inp_path,'dir'))
    file = [inp_path '/' inp_file];
    if (exist(file,'file'))
        try
            [inp_signal, Fs_inp] = audioread(file);
        catch
            [inp_signal, Fs_inp, n_bits] = wavread(file);
        end
    else
        disp('Error: Input file does not exist');
        return;
    end
else
    disp('Error: Input path does not exist');
    return;
end

len_sig = size(inp_signal,1); % in samples

%% Resampling of HRTF to match output sampling frequency

% output sampling frequency is same as input signal sampling frequency
fs = Fs_inp;

% resample the DB if it does not match the input signal sampling frequency
if (Fs_inp ~= fs_db)
    
    if verbose >= 2
    disp('Resampling HRTF database');
    end
    
    % resampling for each channel separately
    hrtf_l = resample(hrtf_l, Fs_inp, fs_db);
    hrtf_r = resample(hrtf_r, Fs_inp, fs_db);
    
end

% length of the impulse response
len_hn = size(hrtf_r,1);

hrir_db(:,:,1) = hrtf_l; %  left channel
hrir_db(:,:,2) = hrtf_r; % right channel

clear hrtf_l hrtf_r;


%% chunk selection parameters
if verbose >= 2
disp('audio segment');
end

% compute the total angular distance of the ideal output file @ specified speed
ang_dist = tot_duration * speed; % in degrees

% compute the number of chunks needed to generate o/p - 1 chunk per position
num_chunks = round(ang_dist/angular_res);

% compute duration (per chunk of data) extracted per spatial position
dur_per_chunk = tot_duration/num_chunks; % in sec

% compute chunk size in samples
orig_chunk_size = ceil(dur_per_chunk * fs);

% output speed computation
out_speed = ang_dist/(num_chunks * orig_chunk_size/fs);

% output duration computation
out_duration = ang_dist/out_speed;



%% trajectory computation
if verbose >= 2
disp('trajectory path');
end

% compute spatial positions at which the response needs to be computed.

% compute the minimum angular resolution supported at this specified speed
min_ang_res = speed / fs; % in degrees

% start and stop azimuth positions of each trajectory supported here
trajectory_azi = [ ...
    azi_L_extreme_pos             0 - min_ang_res;      % trajectory 1
    0             azi_R_extreme_pos - min_ang_res;      % trajectory 2
    azi_R_extreme_pos             0 + min_ang_res;      % trajectory 3
    0             azi_L_extreme_pos + min_ang_res;      % trajectory 4
    ];

% compute the angular distance on each trajectory
if  std(abs(diff(trajectory_azi'))) == 0
    shuttle_dis = mean(abs(diff(trajectory_azi')));
else
    disp('Unsupported: Angular distance covered in each trajectory has to be identical');
    return;
end

% number of different trajectories supported
num_trajectory = length(trajectory_list);

% compute the number of shuttles to take to complete the total duration
num_shuttles = ceil(ang_dist/ceil(shuttle_dis));

% initialize
arr_azi = [];

for k = 1:num_shuttles
    
    % circular addressing
    m = mod(k,num_trajectory);
    if m == 0
        m = num_trajectory;
    end
    
    % shuttle selection
    shuttle = trajectory_list(m);
    
    % extract the start and stop positions for current shuttle
    start_azi = trajectory_azi(shuttle,1);
    stop_azi = trajectory_azi(shuttle,2);
    jump = angular_res * sign(stop_azi - start_azi);
    
    % array updation depending on trajectory
    arr_azi = [arr_azi start_azi:jump:stop_azi];
    
end

% truncate to computed number of chunks
arr_azi = arr_azi(1:num_chunks);

%% sigmoid transition of anuglar position (instead of step) at lower resolution

if flag_sigmoid_trans == 1
    
    % overide the settings to disable cross fade
    flag_enable_xfade = 0;
    
    if angular_res<=1
        % at high angular resolution (lesser than 1 deg). No need
        array_azi = arr_azi;
        
        % chunk size is unaltered
        chunk_size = orig_chunk_size;
        
    else
        % enable Only at low angular resolution (greater than 1 deg)
        
        % angular resolution during sigmoid transition
        ang_res_sigmoid = 1;
        
        % fix the chunk size to requested block size during sigmoid transition
        min_chunk_size = ceil(blk_dur_sgmd * fs);
        
        array_azi = [];
        
        % compute number of blocks at each angular position
        blocks_per_chunk = round(orig_chunk_size/min_chunk_size);

        % compute the number of blocks to be used for sigmoid transition
        blks_per_trans = ceil(angular_res/ang_res_sigmoid);

        % compute the length of the repetition for the "pre-part"
        pre_rep_no = floor((blocks_per_chunk - blks_per_trans)/2);

        % compute the length of the repetition for the "post-part"
        post_rep_no = ceil((blocks_per_chunk - blks_per_trans)/2);

        % do pre-part of first chunk separately as it does not have transition
        array_azi = [array_azi repmat(arr_azi(1), 1, floor(blocks_per_chunk/2))];

        for k=1:num_chunks-1
            % start azimuth value for transition
            start_azi = arr_azi(k);
            
            % stop azimuth value for transition
            stop_azi = arr_azi(k+1);
            
            % transition amount and the direction
            jump = ang_res_sigmoid * sign(stop_azi - start_azi);

            % append the following to the trajectory
                % the fixed angular pre-position 
                % the sigmoid transitionary 
                % the fixed angular post-position
            array_azi = [array_azi                  ...
                repmat(start_azi, 1, pre_rep_no)    ...
                (start_azi:jump:stop_azi-jump)      ...
                repmat(stop_azi, 1, post_rep_no)    ];
            
        end

        % do post-part of last chunk separately as it does not have transition
        array_azi = [array_azi repmat(arr_azi(num_chunks), 1, ...
            ceil(blocks_per_chunk/2))];
        
        % set the block duration to the value used during sigmoid transition
        blk_dur = blk_dur_sgmd;
        
        % set the chunk size to match the value used in sigmoid code
        chunk_size = min_chunk_size;

    end
    
else
    % code when this module is disabled
    array_azi = arr_azi;
    
    % chunk size is unaltered
    chunk_size = orig_chunk_size;
end

% number of spatial positions to concatenate the response
num_azi = length(array_azi);



%% Diffuse field response computation
if verbose >= 2
disp('Diffuse Field Response');
end

% compute indices corresponding to limits of the spectrum for equalization
ind_fmin = ceil(f_min/fs*len_hn); % index corresponding to lower limit
ind_fmax = ceil(f_max/fs*len_hn); % index corresponding to upper limit

N_FFT = 2^nextpow2(len_hn); % for FFT calculation

% for each channel
for ch = 1:n_ch
    
    % convert angle into database index to extract zenith HRTF data
    indx = get_db_indx(180,90);
    
    % extract HRIR
    h_n = hrir_db(:,indx,ch);
    
    % initiate global value with zenith HRTF magnitude
    dfr = (abs(fft(h_n,N_FFT))).^2;
    
    % iterate over every elevation angle
    for ele = first_ele:jump_ele:last_ele % degrees
        
        % iterate over every azimuth angle
        for azi = first_azi:jump_azi:last_azi % degrees
            
            % convert angle into database index to extract data
            indx = get_db_indx(azi,ele);
            
            % extract HRIR
            h_n = hrir_db(:,indx,ch);
            
            % compute HRTF magnitude
            hrtf_mag = (abs(fft(h_n,N_FFT))).^2;
            
            % update global value
            dfr = dfr + hrtf_mag;
            
        end
    end
    
    % RMS computation for Diffuse Field Response
    df_res = (dfr / n_hrir).^0.5;
    dif_fld_res(:,ch) = df_res;
    
    % compute the max gain required within the audible range
    max_gain_lim = 1/min(df_res(ind_fmin:ind_fmax));
    
    % gain limit the equalization filter and store for later processing
      df_eq = min(max_gain_lim,1./df_res);
      dif_fld_eq(:,ch) = df_eq;
    
end


%% HRIR interpolation
if verbose >= 2
disp('HRIR interpolation');
end

% compute the list of unique angular positions
uniq_azi_arr = unique(array_azi);

% compute the total number of unique positions. Optimized memory allocation
num_uniq_azi = length(uniq_azi_arr);

% variable initialization
hrir = zeros(len_hn, num_uniq_azi, n_ch);

% for every azimuth angle of interest. and fixed elevation
for k = 1:num_uniq_azi
    
    azi = uniq_azi_arr(k);
    
    % check if this azimuth is part of the database
    exst = find(azi_list==azi,1);

    if exst
        % data extraction
        indx = get_db_indx(azi,elevation);

        % for each channel
        for ch = 1:n_ch

            % extract HRIR
            h_n = hrir_db(:,indx,ch);

            % store into temp var
            hrir(:,k,ch) = h_n;
        end
    else
        %% interpolation of HRTFs
        azi_a = floor(azi/10)*10;
        azi_b = azi_a + jump_azi;

        % extract indx after converting to angle that conforms with database
        a_indx = get_db_indx(conv_to_db_angle(azi_a),elevation);
        b_indx = get_db_indx(conv_to_db_angle(azi_b),elevation);

        if strcmp(domain,'freq')
        % frequency domain interpolation
            
            % for each channel
            for ch = 1:n_ch
                
                % load data points
                dat_a = hrir_db(:,a_indx,ch);
                dat_b = hrir_db(:,b_indx,ch);
                
                N_FFT = 2^nextpow2(len_hn); % len; % for FFT calculation
                
                % transform the HRIR to HRTF
                fft_a = fft(dat_a, N_FFT);
                fft_b = fft(dat_b, N_FFT);
                
                % obtain the envelope of the spectrum
                spec_a = abs(fft_a); % *2/N_FFT;
                spec_b = abs(fft_b); % *2/N_FFT;
                
                % interpolation of amplitude
                env_o = interp1([azi_a azi_b],[spec_a spec_b]',azi,method);
                
                % interpolation for phase using angles
                angle_a = unwrap(angle(fft_a));
                angle_b = unwrap(angle(fft_b));
                angle_o = interp1([azi_a azi_b],[angle_a angle_b]',azi,method);
                phase_o = exp(+1j .* angle_o);
                
                % inverse tranform on the output
                fft_out = env_o .* phase_o ;
                dat_i =  ifft(fft_out, N_FFT, 'symmetric');
                if isreal(dat_i)==0
                    disp(['Problem at azi: ' num2str(azi) ' deg']);
                end
                
                % store into temp var
                hrir(:,k,ch) = dat_i(1:len_hn);
            end
        elseif strcmp(domain,'time')
        % time domain interpolation
            
            % for each channel
            for ch = 1:n_ch
                
                % load data points
                dat_a = hrir_db(:,a_indx,ch);
                dat_b = hrir_db(:,b_indx,ch);
                
                % interpolation in time domain
                dat_i = interp1([azi_a azi_b],[dat_a dat_b]',azi,method);

                % store into temp var
                hrir(:,k,ch) = dat_i;
            end
        else
            disp('Unsupported: Domain of interpolation');
            return;
        end
    end
end



%% global scale-factor computation
if verbose >= 2
disp('global scale-factor');
end

% intializations
scale_factor = 1;

% variable initialization
hrir_dfr_eq = zeros(len_hn, num_uniq_azi, n_ch);

% compute global scale factor that preserves ILD differences
% over each channel
for ch = 1:n_ch

    % extract diffuse field response
    df_eq = dif_fld_eq(:,ch);

    % at every azimuth angle of interest. and fixed elevation
    for k = 1:num_uniq_azi
    
        % get current azimuth
        azi = uniq_azi_arr(k);
    
        % extract HRIR
        h_n = hrir(:,k,ch);

        % apply dif field equalization
        hn_eq = ifft((fft(h_n,N_FFT) .* df_eq),'symmetric');
        
        % store the dfr equalized HRIR for later use
        hrir_dfr_eq(:,k,ch) = hn_eq(1:len_hn);

        % compute scale factor
        % scale_factor = sum(abs(h_n)); % L1 Norm method
        % scale_factor = max(abs(h_n)); % Chebychev Norm method
        scale_fac = (sum(hn_eq.^2)).^0.5; % L2 Norm method

        % update global scale factor
        scale_factor = max(scale_fac,scale_factor);
    end
end



%% convolution with input signal and concatenation
if verbose >= 1
disp('Computing final output signal');
end

if flag_debug
    if ~(exist(tmp_path,'dir'))
        mkdir(tmp_path);
    end
end

% compute the length of each block
blk_len = ceil(blk_dur * fs); % in samples

% ensure that a block is not bigger than a chunk
if blk_len > chunk_size
    blk_len = chunk_size;
end

% calculate the length of the prev buffer for overlap-and-add
ola_buf_len = max(blk_len, len_hn-1);

% variable initializations
out_sig = zeros(0,n_ch_out);
prev_ola_blk = zeros(ola_buf_len, n_ch_out);
resp_blk = zeros(blk_len,n_ch_out);
m = 0;
chunk_cur_len = 0;

% initializations for cross fading 
if flag_enable_xfade == 1
    % default cross fade length is 1 block. 
    xfade_len = blk_len; % different value is unsupportted
    
    % generate fade in and fade out coefficients
    xfade_sf_cur = linspace(0,1,blk_len)';
    xfade_sf_prv = linspace(1,0,blk_len)';
    
    % initialization for cross fade
    fade_out_blk = [];
    xfading_blk = 0;

else
    % disable cross fading if requested
    xfade_len = 0;
end

% for every azimuth angle of interest. % note: elevation is fixed
% overlap and add method is implemented
for k = 1:num_azi
    
    % get current azimuth
    azi = array_azi(k);
    
    if verbose >= 2
    fprintf('at azimuth: %d deg\r', azi);
    end
    
    % get index corresponding to this azimuth
    ind = find(uniq_azi_arr == azi,1,'first');

    % in debug : to make the HRTF a single filter w/o updates
    if 0 && flag_debug
        ind = find(uniq_azi_arr == 0,1,'first');
    end

    % check whether to process for current angular position or move to next
    while (chunk_cur_len < chunk_size + xfade_len)
        
        if (chunk_cur_len >= chunk_size)
            xfading_blk = 1;
        else
            xfading_blk = 0;
        end
        
        % compute the postions on input signal to obtain the response
        blk_start = 1 + (m * blk_len);
        blk_stop = blk_start + blk_len - 1 ;

        % wrapping of chunk pointer to beginning of file
        if ((blk_start >= len_sig) || (blk_stop > len_sig))
            m = 0;
            blk_start = 1;
            blk_stop = blk_len;
        end
        
        % extract the input signal block of data
        inp_signal_blk = inp_signal(blk_start:blk_stop, :);
        
        % for each channel
        for ch = 1:n_ch_out
            
            % extract diffuse field response equalized HRIR for current position
            if n_ch_out==2
                hrir_eq = hrir_dfr_eq(:,ind,ch);
            elseif n_ch_out==1
                hrir_eq = mean(hrir_dfr_eq(:,ind,:),3);
            end
            
            % convolve with HRIR & apply global scale factor to avoid overflow
            curr_ola_blk = conv(hrir_eq, inp_signal_blk) / scale_factor;
            
            % compute current frame output & overlap with previous frame
            resp_blk(:,ch) = prev_ola_blk(1:blk_len,ch) + curr_ola_blk(1:blk_len);
            
            if xfading_blk==0
                % extract the part of output that overlaps with next frame
                if blk_len == ola_buf_len
                    prev_ola_blk(:,ch) = [curr_ola_blk(blk_len+1:end); ...
                        zeros(blk_len-len_hn+1,1)]; % zero pad
                else
                    prev_ola_blk(:,ch) = curr_ola_blk(blk_len+1:end) + ...
                        [prev_ola_blk(blk_len+1:end,ch); zeros(blk_len,1)];
                end
            end
            
        end
        
        % Debug code: write response as wav file
        if flag_debug
            % tmp_file = [subj_list{subj_no} '_el_' num2str(elevation) '_az_' num2str(arr_azi(k)) ];
            tmp_file = [subj_name '_blk_' num2str(k) ];
            try
                audiowrite([tmp_path '/' tmp_file file_extn], resp_blk, fs);
            catch
                wavwrite(resp_blk, fs, n_bits, [tmp_path '/' tmp_file file_extn]);
            end
        end
        
        % block concatenation
        if flag_enable_xfade == 0
            
            % concatenate the current block
            out_sig = [out_sig; resp_blk];
            
        elseif flag_enable_xfade == 1

            if xfading_blk==0

                % cross fade only when we have data from previous position
                if ~isempty(fade_out_blk)

                    % perform cross fade between same block of data processed 
                    % from previous & current angular positions
                    resp_blk = fade_out_blk .* repmat(xfade_sf_prv,1,2) + ...
                                resp_blk .* repmat(xfade_sf_cur,1,2);

                    % flush the data buffer to inhibit any crossfade
                    fade_out_blk = [];
                end

                % concatenate the current block
                out_sig = [out_sig; resp_blk];

            else
                
                % store current o/p block for cross fading in the next block
                fade_out_blk = resp_blk;
            
            end
            
        end
        
        % update the counter on amount of data produced for current chunk
        chunk_cur_len = chunk_cur_len + blk_len;
        
        % pointer increment, but ensure overlay for cross fade
        if xfading_blk==0
            m = m + 1;
        end
    end
    
    chunk_cur_len = 0;
end



%% save single output file with concatenation across positions

if ~(exist(out_path,'dir'))
    mkdir(out_path);
end

if verbose >= 1
disp ' ';
disp(['Speed requested: ' num2str(speed) ' Actual: ' num2str(out_speed) ' dps']);
disp(['Duration requested: ' num2str(tot_duration) ' Actual: ' num2str(out_duration) ' s']);
end

% concoct output file name
out_file_name = [out_path '/' op_filename_prefix subj_name '_speed_' ...
    num2str(speed) 'dps_' 'path_' sprintf('%d',trajectory_list) ...
    '_res_' num2str(angular_res) 'deg' file_extn];

% write output as wav file
disp ' ';
h_file = fopen(out_file_name,'w');
if h_file>0
    err = fclose(h_file);
    if ~err
        try
  	    audiowrite(out_file_name, out_sig, fs);
        catch
	    wavwrite(out_sig, fs, n_bits, out_file_name);
        end
    end
    if verbose >= 1
    disp('Done. Moving stimulus output saved');
    end
else
    disp('Aborting: Output file is open in another application');
end
disp(['O/p File: ' out_file_name]);

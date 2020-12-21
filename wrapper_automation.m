% This script is used in the automation for the synthesizes of stereo audio
% signals that simulate moving sound sources using Wisconsin HRTF Database
%
% Author:- Pradeep Dheerendra, Auditory Research Group,
% Institute of Neuroscience, Newcastle University, NCL, UK
% (C) copyright reserved 2015

clear all; close all; clc

% set this flag to 1 to use this via wrapper
enable_automation = 1;

% specify the path to the datasets
db_path = 'dataset/';

% specify the path to the input file
inp_path = 'input/';

% specify the path to the output file
output_path = 'outputs_v7p8/';

% specify the file to obtain the response
inp_file = 'modulated_am_noise_80_fs48k_chop.wav';

% specify subject number from 1 to 5
subj_no_list = [1 2 3 4 5];

% specify the speed in degrees per second
speed_list = [25 50 100]; % deg/s

% specify the angular resolution at which to move the sound source
angular_res_list = [1 10];  % in degrees
    
% specify the auditory object trajectory
% note: make a closed loop to avoid sudden jumps
%   1: Left   to Center
%   2: Center to Right
%   3: Right  to Center
%   4: Center to Left
trajectory_list_cell = {[1 4], [4 1], [2 3], [3 2]};

% iterate for all subjects requested
for subj_no = subj_no_list
    
    % specify the path to store the results
    out_path = [output_path '/' num2str(subj_no) '/'];

    % iterate for all angular speeds requested
    for speed = speed_list
        
        % iterate for all angular resoutions requested
        for angular_res = angular_res_list
            
            % iterate for all angular trajectories requested
            for traj = trajectory_list_cell
                
                trajectory_list = traj{1};
                
                % call to the actual program that implements the method
                virtual_moving_sound_synthesis_wisc_v7p8.m;
                
            end
        end
    end
end

% EOF
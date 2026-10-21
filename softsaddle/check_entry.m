function check_entry
% One-batch SoftSaddle call. check.sh prepares work/iniConfig.m first.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'upstream', 'source_code'));
addpath(fullfile(here, 'lib'));
cd(fullfile(here, 'work'));
SoftSaddle
end

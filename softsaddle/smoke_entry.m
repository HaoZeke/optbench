function smoke_entry
% One-batch SoftSaddle call. smoke.sh prepares work/iniConfig.m first.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'upstream', 'source_code'));
addpath(fullfile(here, 'lib'));
cd(fullfile(here, 'work'));
SoftSaddle
end

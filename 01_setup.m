% run at the start of every analysis run 
% central configuration for gender-stroop SPM analysis

%% --- paths ---------------------------------------
cfg.bids_dir   = 'C:\Users\zoele\Desktop\case_studies_project\ds002785';
cfg.deriv_dir  = fullfile(cfg.bids_dir, 'derivatives', 'fmriprep');
cfg.results    = 'C:\Users\zoele\Desktop\case_studies_project\results_spm';
cfg.first_lev  = fullfile(cfg.results, 'first_level');
cfg.second_lev = fullfile(cfg.results, 'second_level');
cfg.masks_dir  = fullfile(cfg.results, 'masks');
cfg.figures    = fullfile(cfg.results, 'figures');

% --- create folder to save results-----------------------
folders = {cfg.results, cfg.first_lev, cfg.second_lev, ...
    cfg.masks_dir, cfg.figures};
for i = 1:length(folders)
    if ~exist(folders{i}, 'dir')
        mkdir(folders{i});
        fprintf('Created: %s\n', folders{i})
    end
end
%% 
% --- scan parameters (can be found in BIDS JSON file)------------
cfg.TR        = 2.0;    % repetition time in seconds
cfg.n_vols    = 245;    % number of volumes per run
cfg.hpf       = 128;    % high-pass filter cutoff in second
%% 
% --- participants
% read participants.tsv
ptable = readtable(fullfile(cfg.bids_dir, 'participants.tsv'), ...
    'Filetype', 'text', 'Delimiter', '\t')

% clean up subjects ID (remove 'sub-' prefix from name)
ptable.subject = strrep(ptable.participant_id, 'sub-', '');

% Find subjects who actually have a BOLD file on disk
bold_files = dir(fullfile(cfg.deriv_dir, 'sub-*', 'func', ...
    '*task-gstroop*MNI152NLin2009cAsym*bold.nii'));
bold_subs = unique(cellfun(@(f) f(5:8), {bold_files.name}, ...
    'UniformOutput', false))';

% Find subjects who actually have an events file on disk
events_files = dir(fullfile(cfg.bids_dir, 'sub-*', 'func', ...
    '*task-gstroop*events.tsv'));
events_subs = unique(cellfun(@(f) f(5:8), {events_files.name}, ...
    'UniformOutput', false))';

% Valid subjects = have BOTH bold and events
valid_subs = intersect(bold_subs, events_subs);

% Filter participants table to valid subjects only
keep   = ismember(ptable.subject, valid_subs);
ptable = ptable(keep, :);


% separation by sex
cfg.males   = ptable.subject(strcmp(ptable.sex, 'M'));
cfg.females = ptable.subject(strcmp(ptable.sex, 'F'));
cfg.all_subs = ptable.subject;
cfg.ptable   = ptable;

%% 

fprintf('\n=== Setup complete ===\n')
fprintf('Total subjects : %d\n', height(ptable))
fprintf('Males          : %d\n', length(cfg.males))
fprintf('Females        : %d\n', length(cfg.females))
fprintf('TR             : %.1f s\n', cfg.TR)
fprintf('Volumes        : %d\n', cfg.n_vols)

% Sanity check
fprintf('\nSubjects with BOLD     : %d\n', length(bold_subs))
fprintf('Subjects with events   : %d\n', length(events_subs))
fprintf('Valid (have both)      : %d\n', length(valid_subs))
% read and inspect the event files for the gstroop files
% understand the timing structure before feeding to the GLM
% converts the seconds to the relative timing of the brain slices scans

run('C:\Users\zoele\Documents\MATLAB\01_setup.m')
%% --- read one example events file -----------------------------------
sub = cfg.all_subs{1};    % take first subject as example

events_file = fullfile(cfg.bids_dir, ...
    sprintf('sub-%s', sub), 'func', ...
    sprintf('sub-%s_task-gstroop_acq-seq_events.tsv', sub));

events = readtable(events_file, 'FileType', 'text', 'Delimiter', '\t');

fprintf('\n=== Events file for sub-%s ===\n', sub)
fprintf('Columns: %s\n', strjoin(events.Properties.VariableNames, ', '))
fprintf('Total trials: %d\n', height(events))
disp(events(1:10, :))  % show first 10 rows
%% --- Check condition counts -----------------------------------------
cond_names = unique(events.trial_type);
fprintf('\nConditions found:\n')
for i = 1:length(cond_names)
    n = sum(strcmp(events.trial_type, cond_names{i}));
    fprintf('  %s : %d trials\n', cond_names{i}, n)
end
%% --- Check timing ---------------------------------------------------
fprintf('\nTiming info:\n')
fprintf('  First trial onset : %.3f s\n', min(events.onset))
fprintf('  Last trial onset  : %.3f s\n', max(events.onset))
fprintf('  Trial duration    : %.3f s\n', mean(events.duration))
fprintf('  Total scan time   : %.1f s (%.0f vols x %.1f s TR)\n', ...
    cfg.n_vols * cfg.TR, cfg.n_vols, cfg.TR)

%% --- Convert onsets from seconds to scans ---------------------------
% SPM needs onsets in SCANS not seconds
% Formula: onset_scans = onset_seconds / TR

con_idx = strcmp(events.trial_type, 'congruent');
inc_idx = strcmp(events.trial_type, 'incongruent');

onsets_con_secs  = events.onset(con_idx);
onsets_inc_secs  = events.onset(inc_idx);
duration_secs    = mean(events.duration);  % same for all trials

onsets_con_scans = onsets_con_secs / cfg.TR;
onsets_inc_scans = onsets_inc_secs / cfg.TR;
duration_scans   = duration_secs   / cfg.TR;

fprintf('\nConverted to scans (dividing by TR=%.1f):\n', cfg.TR)
fprintf('  Congruent onsets (first 5)   : ')
fprintf('%.2f  ', onsets_con_scans(1:5)); fprintf('\n')
fprintf('  Incongruent onsets (first 5) : ')
fprintf('%.2f  ', onsets_inc_scans(1:5)); fprintf('\n')
fprintf('  Duration in scans            : %.3f\n', duration_scans)
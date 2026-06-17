%% 03_first_level.m
% Runs the first-level GLM for each subject
% For each subject this script:
%   1. Finds their BOLD file and events file
%   2. Builds a design matrix (the mathematical model)
%   3. Fits the GLM
%   4. Computes two contrasts:
% - incongruent > congruent (Stroop interference - H1, H3)
% - task > baseline (overall activation - H2)

run('C:\Users\zoele\Documents\MATLAB\Scripts\01_setup.m')


%% --- Loop over all subjects -----------------------------------------

spm_jobman('initcfg')           % intialize SPM batch 
spm('defaults', 'fmri')         % load default parameters for fmri analysis
skipped = {};                   % collect ID numbers of participants that cannot be analyzed

for s = 1:length(cfg.all_subs)
    sub = cfg.all_subs{s};             % subject ID
    fprintf('\n[%d/%d] sub-%s ... ', s, length(cfg.all_subs), sub)


     %  insert the subject ID into filename templates: 
     % where results will be saved (sub_dir)
     % where preprocessed BOLD file lives,
     % and where events TSV lives.
    sub_dir   = fullfile(cfg.first_lev, sprintf('sub-%s', sub));
    bold_path = fullfile(cfg.deriv_dir, sprintf('sub-%s', sub), 'func', ...
        sprintf('sub-%s_task-gstroop_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii', sub));
    events_file = fullfile(cfg.bids_dir, sprintf('sub-%s', sub), 'func', ...
        sprintf('sub-%s_task-gstroop_acq-seq_events.tsv', sub));

    % Check all required files exist and not already processed
    already_done   = exist(fullfile(sub_dir, 'SPM.mat'), 'file');
    has_bold       = exist(bold_path, 'file');
    has_events     = exist(events_file, 'file');

    if already_done
        fprintf('already done, skipping\n')

    elseif ~has_bold
        fprintf('BOLD not found, skipping\n')
        skipped{end+1} = sub;

    elseif ~has_events
        fprintf('Events not found, skipping\n')
        skipped{end+1} = sub;

    else
        % If all files present - proceed with GLM       
        if ~exist(sub_dir, 'dir'), mkdir(sub_dir); end

        % -- Build volume list of all the files (245 instead of one 4D
        % scan)
        scans = cell(cfg.n_vols, 1);
        for v = 1:cfg.n_vols
            scans{v} = sprintf('%s,%d', bold_path, v);
        end

        % -- Read events --
        events  = readtable(events_file, 'FileType', 'text', 'Delimiter', '\t');
        con_idx = strcmp(events.trial_type, 'congruent');
        inc_idx = strcmp(events.trial_type, 'incongruent');
        onsets_con  = events.onset(con_idx) / cfg.TR;           % convert onsets from seconds to scans
        onsets_inc  = events.onset(inc_idx) / cfg.TR;
        duration    = mean(events.duration)  / cfg.TR;          % average trial duration (all trials have the same duration)

        % -- Load confounds --
        confounds_file = fullfile(cfg.deriv_dir, sprintf('sub-%s', sub), 'func', ...
            sprintf('sub-%s_task-gstroop_acq-seq_desc-confounds_regressors.tsv', sub));
        confound_cols = {'trans_x', 'trans_x_derivative1', 'trans_x_power2', ...
    'trans_x_derivative1_power2', 'trans_y', 'trans_y_derivative1', ...
    'trans_y_power2', 'trans_y_derivative1_power2', 'trans_z', ...
    'trans_z_derivative1', 'trans_z_power2', 'trans_z_derivative1_power2', ...
    'rot_x', 'rot_x_derivative1', 'rot_x_power2', 'rot_x_derivative1_power2', ...
    'rot_y', 'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2', ...
    'rot_z', 'rot_z_derivative1', 'rot_z_power2', 'rot_z_derivative1_power2', ...
    'csf', 'white_matter', 'framewise_displacement'};
        R = [];
        if exist(confounds_file, 'file')
            conf_table = readtable(confounds_file, 'FileType', 'text', 'Delimiter', '\t');
            for c = 1:length(confound_cols)
                col = confound_cols{c};
                if ismember(col, conf_table.Properties.VariableNames)
                    vals = conf_table.(col);
                    vals(isnan(vals)) = 0;
                    R = [R, vals];
                end
            end
        end

        % -- Build matlabbatch --
        matlabbatch = {};

        % Model specification - build design matrix (same options you can
        % use in SPM GUI)
        matlabbatch{1}.spm.stats.fmri_spec.dir                    = {sub_dir};  % directory
        matlabbatch{1}.spm.stats.fmri_spec.timing.units           = 'scans';    % units are in scans instead of seconds
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT              = cfg.TR;     % repetition time (2.0 seconds)
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t          = 16;         % microtime resolution settings
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0         = 8;          % reference bin (8)
        matlabbatch{1}.spm.stats.fmri_spec.sess.scans             = scans;      % 245-element cell array of frame references build 

        % Congruent condition
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name     = 'congruent';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset    = onsets_con;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = duration;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod     = 0;           % no time modulation - not needed
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod     = struct('name',{},'param',{},'poly',{});  % parametic modulation - empty because not needed
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth     = 1;           % no modulators - so no effect but has to be included

        % Incongruent condition
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name     = 'incongruent';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset    = onsets_inc;  % vector computed earlier
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = duration;    % vector computed earlier
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod     = 0;           % no time-modulation 
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod     = struct('name',{},'param',{},'poly',{});  % idem as before
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth     = 1;           % idem as before

        % Confound regressors
        matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name',{},'val',{});       % add regressors of no interest (27 extra columns)
        if ~isempty(R)
            for c = 1:size(R,2)
                matlabbatch{1}.spm.stats.fmri_spec.sess.regress(c).name = confound_cols{c};
                matlabbatch{1}.spm.stats.fmri_spec.sess.regress(c).val  = R(:,c);
            end
        end

        % Other settings
        matlabbatch{1}.spm.stats.fmri_spec.sess.hpf         = cfg.hpf;      % high-pass filter cut-off
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];        % use canonical hemodynamic response function 
        matlabbatch{1}.spm.stats.fmri_spec.volt             = 1;            % no non-linear interaction function 
        matlabbatch{1}.spm.stats.fmri_spec.global           = 'None';       % no global intensity normalization 
        matlabbatch{1}.spm.stats.fmri_spec.mthresh          = 0.8;          % implicit masking treshold (voxels whose mean intensity is below 80% of the global mean are excluded from analysis)
        matlabbatch{1}.spm.stats.fmri_spec.mask             = {''};         % no explicit mask usage
        matlabbatch{1}.spm.stats.fmri_spec.cvi              = 'AR(1)';      % standard correction since fMRI timepoints are statistically dependent

        % Model estimation
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1)        = cfg_dep( ...    % fits the GLM at every voxel - creates dependency to use model that previous module produces
            'fMRI model specification: SPM.mat File', ...
            substruct('.','val','{}',{1},'.','val','{}',{1},'.','val','{}',{1}), ...   
            substruct('.','spmmat'));
        matlabbatch{2}.spm.stats.fmri_est.write_residuals  = 0;                % don't save voxel wise residuals images
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;                % use classical parameter estimation via restricted maximum likelihood

        % Contrasts
        matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep( ...                   % same dependency but pointing at module 2
            'Model estimation: SPM.mat File', ...
            substruct('.','val','{}',{2},'.','val','{}',{1},'.','val','{}',{1}), ...
            substruct('.','spmmat'));

        matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = 'inc_vs_con';
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1];          %  contrast: (+1 × congruent_beta) + (-1 × incongruent_beta)
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

        matlabbatch{3}.spm.stats.con.consess{2}.tcon.name    = 'task_vs_baseline';
        matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [0.5 0.5];       % average activation for incongruent and congruent
        matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';          % no replication across muliple sessions

        matlabbatch{3}.spm.stats.con.delete = 1;            % delete existing contrasts before adding new ones

        % Run
        try
            spm_jobman('run', matlabbatch)
            fprintf('done\n')
        catch ME
            fprintf('ERROR: %s\n', ME.message)  % print error message, record subject, continue to next one
            skipped{end+1} = sub;
        end

    end  % closes main else block
end  % closes for loop
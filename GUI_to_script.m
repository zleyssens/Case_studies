% this script was used to generate the script equivalent to the batch GUI 

out_path = fullfile(cfg.results, 'second_level_batch_GUI_equivalent.m');
fid = fopen(out_path, 'w');

fprintf(fid, '%% Second-level GLM batch script\n');
fprintf(fid, '%% Equivalent to what was done through SPM GUI\n');
fprintf(fid, '%% Generated: %s\n\n', datestr(now));
fprintf(fid, 'addpath(''C:\\Users\\zoele\\Documents\\MATLAB\\spm'')\n');
fprintf(fid, 'spm(''defaults'', ''FMRI'')\n');
fprintf(fid, 'spm_jobman(''initcfg'')\n\n');

hypotheses = {
    fullfile(cfg.second_lev, 'H1_inc_vs_con', 'SPM.mat'), 'H1: One-sample t-test (inc > con)';
    fullfile(cfg.second_lev, 'H2_sex_task',   'SPM.mat'), 'H2: Two-sample t-test (sex differences)';
    fullfile(cfg.second_lev, 'H3_sex_stroop', 'SPM.mat'), 'H3: Two-sample t-test (Stroop by sex)';
};

for h = 1:size(hypotheses, 1)
    spm_path = hypotheses{h, 1};
    label    = hypotheses{h, 2};

    if ~exist(spm_path, 'file')
        fprintf(fid, '\n%%%% %s - file not found\n', label);
        continue
    end

    load(spm_path)

    fprintf(fid, '\n%%%% ================================================\n');
    fprintf(fid, '%%%% %s\n', label);
    fprintf(fid, '%%%% ================================================\n');
    fprintf(fid, 'matlabbatch = {};\n\n');
    fprintf(fid, '%% Step 1: Specify design\n');
    fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.dir = {''%s''};\n\n', ...
        fileparts(spm_path));

    if strcmp(SPM.xsDes.Design, 'One sample t-test')
        fprintf(fid, '%% Design: One-sample t-test (%d subjects)\n', length(SPM.xY.VY));
        fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = {\n');
        for v = 1:length(SPM.xY.VY)
            fprintf(fid, '    ''%s''\n', SPM.xY.VY(v).fname);
        end
        fprintf(fid, '};\n');

    elseif strcmp(SPM.xsDes.Design, 'Two-sample t-test')
        X      = SPM.xX.X;
        g1_idx = find(X(:,1) == 1);
        g2_idx = find(X(:,2) == 1);

        fprintf(fid, '%% Design: Two-sample t-test (Group1=%d, Group2=%d)\n', ...
            length(g1_idx), length(g2_idx));
        fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = {\n');
        for v = g1_idx'
            fprintf(fid, '    ''%s''\n', SPM.xY.VY(v).fname);
        end
        fprintf(fid, '};\n');
        fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = {\n');
        for v = g2_idx'
            fprintf(fid, '    ''%s''\n', SPM.xY.VY(v).fname);
        end
        fprintf(fid, '};\n');
        fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.des.t2.dept     = 0;\n');
        fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;\n');
        fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca    = 0;\n');
        fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova   = 0;\n');
    end

    fprintf(fid, '\n%% Masking\n');
    fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none    = 1;\n');
    fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.masking.im             = 1;\n');
    fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.masking.em             = {''''};\n');
    fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit         = 1;\n');
    fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;\n');
    fprintf(fid, 'matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm        = 1;\n');

    fprintf(fid, '\n%% Step 2: Estimate model\n');
    fprintf(fid, 'matlabbatch{2}.spm.stats.fmri_est.spmmat           = {''%s''};\n', spm_path);
    fprintf(fid, 'matlabbatch{2}.spm.stats.fmri_est.write_residuals  = 0;\n');
    fprintf(fid, 'matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;\n');

    fprintf(fid, '\n%% Step 3: Contrasts\n');
    fprintf(fid, 'matlabbatch{3}.spm.stats.con.spmmat = {''%s''};\n', spm_path);
    for c = 1:length(SPM.xCon)
        fprintf(fid, 'matlabbatch{3}.spm.stats.con.consess{%d}.tcon.name    = ''%s'';\n', ...
            c, SPM.xCon(c).name);
        weights = SPM.xCon(c).c(1:min(2,end))';
        fprintf(fid, 'matlabbatch{3}.spm.stats.con.consess{%d}.tcon.weights = [', c);
        fprintf(fid, '%g ', weights);
        fprintf(fid, '];\n');
        fprintf(fid, 'matlabbatch{3}.spm.stats.con.consess{%d}.tcon.sessrep = ''none'';\n', c);
    end
    fprintf(fid, 'matlabbatch{3}.spm.stats.con.delete = 0;\n');
    fprintf(fid, '\n%% Run the batch\n');
    fprintf(fid, 'spm_jobman(''run'', matlabbatch)\n');
end

fclose(fid);
fprintf('Script saved to:\n%s\n', out_path)
open(out_path)

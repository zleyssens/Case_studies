% Load AAL2
aal_dir = 'C:\Users\zoele\Documents\MATLAB\spm\toolbox\aal';
V_aal   = spm_vol(fullfile(aal_dir, 'ROI_MNI_V5.nii'));
aal_vol = spm_read_vols(V_aal);
xml_str = fileread(fullfile(aal_dir, 'ROI_MNI_V5.xml'));

pattern = '<index>(\d+)</index>\s*<name>(.*?)</name>';
matches = regexp(xml_str, pattern, 'tokens');
aal_labels = struct();
for i = 1:length(matches)
    aal_labels(i).index = str2double(matches{i}{1});
    aal_labels(i).name  = matches{i}{2};
end
fprintf('AAL2 loaded: %d regions\n\n', length(aal_labels))

% Settings
t_threshold = 3.13;
k_min       = 0;

% All contrasts
contrasts = {
    'H1_inc_vs_con',  'spmT_0001.nii', 'H1_inc_vs_con_k0',          'H1: Incongruent > Congruent';
    'H2_sex_task',    'spmT_0001.nii', 'H2_males_gt_females_k0',    'H2: Males > Females (task)';
    'H2_sex_task',    'spmT_0002.nii', 'H2_females_gt_males_k0',    'H2: Females > Males (task)';
    'H3_sex_stroop',  'spmT_0001.nii', 'H3_males_gt_females_k0',    'H3: Males > Females (Stroop)';
    'H3_sex_stroop',  'spmT_0002.nii', 'H3_females_gt_males_k0',    'H3: Females > Males (Stroop)';
};

for ci = 1:size(contrasts, 1)
    h_dir    = contrasts{ci, 1};
    t_fname  = contrasts{ci, 2};
    out_name = contrasts{ci, 3};
    label    = contrasts{ci, 4};

    t_file = fullfile(cfg.second_lev, h_dir, t_fname);
    if ~exist(t_file, 'file')
        fprintf('FILE NOT FOUND: %s\n\n', t_file)
        continue
    end

    V_t   = spm_vol(t_file);
    t_vol = spm_read_vols(V_t);

    bin_vol = t_vol > t_threshold;
    [bin_vol_int, n_clusters] = spm_bwlabel(double(bin_vol), 26);

    clus_sizes = zeros(max(n_clusters, 1), 1);
    for cc = 1:n_clusters
        clus_sizes(cc) = sum(bin_vol_int(:) == cc);
    end

    fprintf('========================================================\n')
    fprintf('%s\n', label)
    fprintf('Threshold: T = %.2f, p < 0.001 uncorrected, k >= %d\n', ...
        t_threshold, k_min)
    fprintf('Peak t = %.3f | Suprathreshold voxels = %d | Clusters = %d\n', ...
        max(t_vol(:)), sum(bin_vol(:)), n_clusters)
    fprintf('========================================================\n')

    if n_clusters == 0
        fprintf('No suprathreshold voxels\n\n')
        continue
    end

    [~, order] = sort(clus_sizes, 'descend');

    fprintf('%-5s %-5s %-8s %-22s %-40s\n', ...
        'Clus', 'k', 'T', 'MNI [x  y  z]', 'AAL2 Region')
    fprintf('%s\n', repmat('-', 1, 85))

    csv_path = fullfile(cfg.results, sprintf('%s_clean.csv', out_name));
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'Cluster,k_voxels,T_peak,x,y,z,Brain_region\n');

    n_shown = 0;
    for oi = 1:length(order)
        cc = order(oi);
        k  = clus_sizes(cc);
        n_shown = n_shown + 1;

        % Find peak voxel in cluster
        cluster_mask      = (bin_vol_int == cc);
        cluster_t         = t_vol .* cluster_mask;
        [peak_t, lin_idx] = max(cluster_t(:));
        [px, py, pz]      = ind2sub(size(t_vol), lin_idx);
        peak_mm           = V_t.mat * [px; py; pz; 1];
        mni               = round(peak_mm(1:3))';

        % AAL label
        vox_aal = round(V_aal.mat \ [mni, 1]');
        vox_aal = vox_aal(1:3);
        dims    = size(aal_vol);

        if any(vox_aal < 1) || any(vox_aal' > dims)
            region = 'Outside brain';
        else
            aal_idx = aal_vol(vox_aal(1), vox_aal(2), vox_aal(3));
            if aal_idx == 0
                region = 'No label (white matter)';
            else
                m = find([aal_labels.index] == aal_idx);
                if ~isempty(m)
                    region = make_readable(aal_labels(m(1)).name);
                else
                    region = sprintf('Unknown %d', aal_idx);
                end
            end
        end

        fprintf('%-5d %-5d %-8.2f [%4d %4d %4d]    %s\n', ...
            n_shown, k, peak_t, mni(1), mni(2), mni(3), region)
        fprintf(fid, '%d,%d,%.2f,%d,%d,%d,%s\n', ...
            n_shown, k, peak_t, mni(1), mni(2), mni(3), region)

        if n_shown >= 30, break; end
    end
    fclose(fid)
    fprintf('\nSaved: %s\n\n', csv_path)
end

fprintf('=== All done ===\n')

function clean = make_readable(raw)
    lookup = {
        'Frontal_Sup_2_R',    'R superior frontal gyrus (dorsolateral)';
        'Frontal_Sup_2_L',    'L superior frontal gyrus (dorsolateral)';
        'Frontal_Mid_2_R',    'R middle frontal gyrus';
        'Frontal_Mid_2_L',    'L middle frontal gyrus';
        'Frontal_Inf_Tri_R',  'R inferior frontal gyrus (pars triangularis)';
        'Frontal_Inf_Tri_L',  'L inferior frontal gyrus (pars triangularis)';
        'Frontal_Inf_Oper_R', 'R inferior frontal gyrus (pars opercularis)';
        'Frontal_Inf_Oper_L', 'L inferior frontal gyrus (pars opercularis)';
        'Frontal_Sup_Medial_R','R superior medial frontal gyrus';
        'Frontal_Sup_Medial_L','L superior medial frontal gyrus';
        'Frontal_Med_Orb_R',  'R medial orbital frontal cortex';
        'Frontal_Med_Orb_L',  'L medial orbital frontal cortex';
        'Supp_Motor_Area_R',  'R supplementary motor area';
        'Supp_Motor_Area_L',  'L supplementary motor area';
        'Precentral_R',       'R precentral gyrus';
        'Precentral_L',       'L precentral gyrus';
        'Postcentral_R',      'R postcentral gyrus';
        'Postcentral_L',      'L postcentral gyrus';
        'Cingulate_Mid_R',    'R middle cingulate cortex';
        'Cingulate_Mid_L',    'L middle cingulate cortex';
        'Cingulate_Ant_R',    'R anterior cingulate cortex';
        'Cingulate_Ant_L',    'L anterior cingulate cortex';
        'Cingulate_Post_R',   'R posterior cingulate cortex';
        'Cingulate_Post_L',   'L posterior cingulate cortex';
        'Precuneus_R',        'R precuneus';
        'Precuneus_L',        'L precuneus';
        'Cuneus_R',           'R cuneus';
        'Cuneus_L',           'L cuneus';
        'Calcarine_R',        'R calcarine cortex';
        'Calcarine_L',        'L calcarine cortex';
        'Lingual_R',          'R lingual gyrus';
        'Lingual_L',          'L lingual gyrus';
        'SupraMarginal_R',    'R supramarginal gyrus';
        'SupraMarginal_L',    'L supramarginal gyrus';
        'Angular_R',          'R angular gyrus';
        'Angular_L',          'L angular gyrus';
        'Parietal_Sup_R',     'R superior parietal lobule';
        'Parietal_Sup_L',     'L superior parietal lobule';
        'Parietal_Inf_R',     'R inferior parietal lobule';
        'Parietal_Inf_L',     'L inferior parietal lobule';
        'Occipital_Sup_R',    'R superior occipital gyrus';
        'Occipital_Sup_L',    'L superior occipital gyrus';
        'Occipital_Mid_R',    'R middle occipital gyrus';
        'Occipital_Mid_L',    'L middle occipital gyrus';
        'Occipital_Inf_R',    'R inferior occipital gyrus';
        'Occipital_Inf_L',    'L inferior occipital gyrus';
        'Fusiform_R',         'R fusiform gyrus';
        'Fusiform_L',         'L fusiform gyrus';
        'Temporal_Sup_R',     'R superior temporal gyrus';
        'Temporal_Sup_L',     'L superior temporal gyrus';
        'Temporal_Mid_R',     'R middle temporal gyrus';
        'Temporal_Mid_L',     'L middle temporal gyrus';
        'Temporal_Inf_R',     'R inferior temporal gyrus';
        'Temporal_Inf_L',     'L inferior temporal gyrus';
        'Temporal_Pole_Sup_R','R superior temporal pole';
        'Temporal_Pole_Sup_L','L superior temporal pole';
        'Temporal_Pole_Mid_R','R middle temporal pole';
        'Temporal_Pole_Mid_L','L middle temporal pole';
        'Hippocampus_R',      'R hippocampus';
        'Hippocampus_L',      'L hippocampus';
        'ParaHippocampal_R',  'R parahippocampal gyrus';
        'ParaHippocampal_L',  'L parahippocampal gyrus';
        'Amygdala_R',         'R amygdala';
        'Amygdala_L',         'L amygdala';
        'Insula_R',           'R insula';
        'Insula_L',           'L insula';
        'Thalamus_R',         'R thalamus';
        'Thalamus_L',         'L thalamus';
        'Caudate_R',          'R caudate';
        'Caudate_L',          'L caudate';
        'Putamen_R',          'R putamen';
        'Putamen_L',          'L putamen';
        'Pallidum_R',         'R pallidum';
        'Pallidum_L',         'L pallidum';
        'Cerebelum_4_5_R',    'R cerebellum (lobules IV-V)';
        'Cerebelum_4_5_L',    'L cerebellum (lobules IV-V)';
        'Cerebelum_6_R',      'R cerebellum (lobule VI)';
        'Cerebelum_6_L',      'L cerebellum (lobule VI)';
        'Cerebelum_7b_R',     'R cerebellum (lobule VIIb)';
        'Cerebelum_7b_L',     'L cerebellum (lobule VIIb)';
        'Cerebelum_8_R',      'R cerebellum (lobule VIII)';
        'Cerebelum_8_L',      'L cerebellum (lobule VIII)';
        'Cerebelum_Crus1_R',  'R cerebellum (crus I)';
        'Cerebelum_Crus1_L',  'L cerebellum (crus I)';
        'Cerebelum_Crus2_R',  'R cerebellum (crus II)';
        'Cerebelum_Crus2_L',  'L cerebellum (crus II)';
        'Vermis_4_5',         'Cerebellar vermis (lobules IV-V)';
        'Vermis_6',           'Cerebellar vermis (lobule VI)';
        'Vermis_7',           'Cerebellar vermis (lobule VII)';
        'Vermis_8',           'Cerebellar vermis (lobule VIII)';
    };
    clean = raw;
    for r = 1:size(lookup, 1)
        if strcmp(raw, lookup{r, 1})
            clean = lookup{r, 2};
            return
        end
    end
end
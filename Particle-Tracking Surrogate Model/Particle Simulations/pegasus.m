close all;
clearvars; 
clc;

%% Define distributions
nsamples            = 200;
Qtots               = 10 * rand(1, nsamples);
gun_loopmvs         = 35 + 5 * rand(1, nsamples);
gun_phasedegs        = 15 + 15 * rand(1, nsamples);
linac_gradients     = 13 + 5 * rand(1, nsamples);
linac_phasedegs     = 210 + 90 * rand(1, nsamples);

% Outputs
time_std            = zeros(1, nsamples); % Femtoseconds
G_mean              = zeros(1, nsamples); 
G_std               = zeros(1, nsamples); % Percent

%% Run Simulations
for j=1:nsamples
    if mod(j, 2)==0
        fprintf('Iter. %d of %d\n', j, nsamples)
    end
    
    %%% Define inputs
    mr.nps              = 500;
    mr.Qtot             = -1e-12 * Qtots(j); %0 to 10pC

    mr.gun_loopmv       = gun_loopmvs(j); %35 to 40
    mr.gun_phasedeg     = gun_phasedegs(j); %15 to 30

    mr.linac_gradient   = linac_gradients(j); %13 to 18
    mr.linac_phasedeg   = linac_phasedegs(j); %210 to 300

    write_mr(mr);

    %%% Run particle-tracking simulation
    env_cells           = split(fileread('.env'), '=');
    GPTLICENSE          = env_cells{2};

    tic;
    gptcall     = sprintf('mr -o beam_after_gun.gdf params.mr gpt pegasus_gun.in GPTLICENSE=%s\n', GPTLICENSE);
    system(gptcall);
    gptcall     = sprintf('mr -o results.gdf params.mr gpt pegasus_transport.in GPTLICENSE=%s\n', GPTLICENSE);
    system(gptcall);
%     fprintf('Simulation done: %.1f sec\n', toc);

    %%% Read/append results
    gdf                 = load_gdf('results.gdf');
    G                   = mean(gdf(1).d.G);
    
    time_stds(j)        = std(gdf(1).d.t)*1e15;
    G_means(j)          = G;
    G_stds(j)           = std(gdf(1).d.G)/G*100;


%     fprintf('G = %.2f\nsigG = %.2f%%\nsigt = %d fs\n', G, sigG, round(sigt));
end

%% Write to csv
datetime_string         = datestr( datetime('now'), 'yyyy-mm-dd_HH-MM-SS');
filename                = ['Simulation Data/', datetime_string, '.csv'];

T   = table(Qtots', gun_loopmvs', gun_phasedegs', linac_gradients', linac_phasedegs',...
    time_stds', G_means', G_stds',...
    'VariableNames', {'charge', 'gun_amp', 'gun_phase', 'linac_amp', 'linac_phase', 'time_std', 'G_mean', 'G_std'});

writetable(T, filename);
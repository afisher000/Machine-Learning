close all;
clearvars; 
clc;

p   = load_pegasus();

%% Define inputs
mr.nps              = 100;
mr.Qtot             = -10e-12;

mr.gun_loopmv       = 39;
mr.gun_phasedeg     = 6*2.856;

mr.sol1_current     = 1.1;

mr.linac_gradient   = 18;
mr.linac_phasedeg   = 150;


write_mr(mr);

%% Run particle-tracking simulation
input_file          = 'pegasus.in';
GPTLICENSE          = 
tic;
gptcall     = sprintf('mr -o %s %s gpt %s GPTLICENSE=%d\n', output_file, mr_file, input_file, GPTLICENSE);
system(gptcall);
fprintf('Simulation done: %.1f sec\n', toc);
    
run_GPT('pegasus.in');

%



key = ''; % key to prepend file names (for separating runs)

% Option to save basis-functions on largest indices
plot_basis = 1;
nb_basis   = 6;

% Option to save LS error 
least_square_error = true;

% Set architecture
blocktype       = 'default'
layers          = 5 
nodes_per_layer = 50
tol             = '5e-07';
opt             = 'Adam';

% run IDs
unique_run_ID   = 'cedar_CAS_set1_1st_arch_he_normal_Mar16'
unique_run_ID_2 = [];

name_run_ID     = unique_run_ID
unique_run_ID_3 = []; 

activations     = ["tanh", "relu", "elu", "sigmoid"];
init_act_num    = 1;
final_act_num   = 2;

if nodes_per_layer == 300 
    max_M_points = 4800;
else
    max_M_points = 5000;
end

dim_vals      = [1 2 4 8 16];
example_vals  = [1 2 3 5];

for ex_num = 1:3 %1:length(example_vals)
    
    example_num = example_vals(ex_num);

    for d_val = 1:2  % 1:length(dim_vals)-1
        
        dim = dim_vals(d_val);

        switch dim
            case 1
                points = 10000
                nb_test_points = 16385 
            case 2
                points = 10000
                nb_test_points = 15361 
            case 4
                points = 20000
            case 8
                points = 50000
            case 16
                points = 100000
            otherwise
                disp('incorrect dim')
        end

        for i = init_act_num:final_act_num
            
            activation = convertStringsToChars(activations(i));
            %TODO: change the base_dir for your own case
            base_dir = '~/scratch/DNN_sampling';
            run_ID = [unique_run_ID '_' activation '_' blocktype '_' num2str(layers) 'x' num2str(nodes_per_layer) '_',...
                        num2str(points,'%06.f') '_pnts_' tol '_tol_' opt '_opt_example_' ,...
                        num2str(example_num) '_dim_' num2str(dim)]
            
            if isempty(unique_run_ID_2) == 0
                %TODO: delete this later, it's purpose is put together different unique_ID runs 
                run_ID_2 = [unique_run_ID_2 '_' activation '_' blocktype '_' num2str(layers) 'x' num2str(nodes_per_layer) '_',...
                            num2str(points,'%06.f') '_pnts_' tol '_tol_' opt '_opt_example_' ,...
                            num2str(example_num) '_dim_' num2str(dim)]
            else 
                run_ID_2 = [];
            end

            samp_mode_dir_names = ["ASGD",...
                                "MC",...
                                "MC2"];

            samp_mode_names = ["ASGD",...
                            "MC",...
                            "MC2"];

            num_modes = size(samp_mode_names,1)

            num_trials = 20;
            num_steps  = 10;

            final_run_ID = [name_run_ID '_' activation '_' blocktype '_' num2str(layers) 'x' num2str(nodes_per_layer) '_',...
                            num2str(points,'%06.f') '_pnts_' tol '_tol_' opt '_opt_example_' ,...
                            num2str(example_num) '_dim_' num2str(dim)]

            output_filename = [final_run_ID '_extracted_data.mat'];

            extracted_data        = matfile(output_filename,'Writable',true);
            L2_error_save_data    = zeros(num_modes, num_trials, num_steps);
            M_values_save_data    = zeros(num_modes, num_trials, num_steps);
            C_fB_values_save_data = zeros(num_modes, num_trials, num_steps);
            C_fQ_values_save_data = zeros(num_modes, num_trials, num_steps);
            x_train_save_data     = zeros(num_modes, num_trials, max_M_points,dim);
            Chris_vals_save_data  = zeros(num_modes, num_trials, points, num_steps);
            Prob_dist_save_data   = zeros(num_modes, num_trials, points, num_steps);
            r_values_save_data    = zeros(num_modes, num_trials, num_steps);

            largest_indices_save_data     = zeros(num_modes, num_trials, nb_basis);
            largest_basis_train_save_data = zeros(num_modes, num_trials, points, nb_basis);
            largest_coefficients          = zeros(num_modes, num_trials, nb_basis);
            
            for samp_mode = 1:2

                samp_mode_name = convertStringsToChars(samp_mode_names(samp_mode));
                samp_mode_dir_name = convertStringsToChars(samp_mode_dir_names(samp_mode));
                
                if isempty(run_ID_2) == 1
                    results_dir = [base_dir '/' run_ID '_training_method_' samp_mode_dir_name] 
                
                else 
                    % TODO: delete this later 
                    if samp_mode == 1 
                        % ASGD sampling
                        results_dir = [base_dir '/' run_ID '_training_method_' samp_mode_dir_name] 
                    elseif samp_mode == 2
                        % MC sampling
                        results_dir = [base_dir '/' run_ID_2 '_training_method_' samp_mode_dir_name] 
                    else
                        disp('wrong samp mode')
                    end
                end

                for t = 1:num_trials

                    filename = ['trial_' num2str(t-1)];
                    filename = [results_dir '/' filename '/run_data.mat']

                    if isfile(filename)
                        data = matfile(filename,'Writable',false);
                        
                        L2_error_save_data(samp_mode,t,:)     = data.L2_error_data;
                        M_values_save_data(samp_mode,t,:)     = data.M_values;
                        
                        C_fB_values_save_data(samp_mode,t,:)  = 0; %data.constant_C_from_B_data;
                        C_fQ_values_save_data(samp_mode,t,:)  = data.constant_C_from_Q_data;
                        
                        x_train_save_data(samp_mode,t,:,:)    = data.x_train_data;
                        Chris_vals_save_data(samp_mode,t,:,:) = data.Chris_func_vals;
                        Prob_dist_save_data(samp_mode,t,:,:)  = data.Prob_dist_vals;
                        r_values_save_data(samp_mode,t,:)     = data.r_values;

                        if (dim == 1 ) && (plot_basis == 1) 
                            largest_indices_save_data(samp_mode,t,:)       = data.largest_indices_coeff;
                            largest_basis_train_save_data(samp_mode,t,:,:) = data.basis_on_largest_coef_train;
                            largest_coeff_save_data(samp_mode,t,:)         = data.largest_coeff; 
                        end

                        
                        if least_square_error
                            l2_error_ls_save_data(samp_mode,t) = data.l2_error_least_squares;
                            L2_error_ls_save_data(samp_mode,t) = data.L2_error_least_squares;
                        end

                    else
                        disp([filename ' is missing']);
                    end

                end

            end

            extracted_data.L2_error_save_data    = L2_error_save_data;
            extracted_data.M_values_save_data    = M_values_save_data;
            extracted_data.C_fB_values_save_data = C_fB_values_save_data;
            extracted_data.C_fQ_values_save_data = C_fQ_values_save_data;
            extracted_data.x_train_save_data     = x_train_save_data;
            extracted_data.r_values_save_data    = r_values_save_data;
            extracted_data.Chris_vals_save_data  = Chris_vals_save_data; 
            extracted_data.Prob_dist_save_data   = Prob_dist_save_data;
            
            if least_square_error
                extracted_data.l2_error_ls_save_data = l2_error_ls_save_data;
                extracted_data.L2_error_ls_save_data = L2_error_ls_save_data;
            end

            if (dim == 1) && (plot_basis == 1)
                extracted_data.largest_indices_save_data     = largest_indices_save_data;
                extracted_data.largest_basis_train_save_data = largest_basis_train_save_data;
                extracted_data.largest_coeff_save_data       = largest_coeff_save_data;
            end

        end
    end
end

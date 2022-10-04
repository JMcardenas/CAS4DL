%-------------------------------------------------------------------------%
% Filename: figs_5_plot.m 
% Author: Juan M. Cardenas, Ben Adcock, Nick Dexter
% Part of the paper "CAS4DL: Christoffel Adaptive Sampling for function 
% approximation via Deep Learning"
%
% Description: generates the plots for Figure 5.
%
% Inputs: 
%
% Outputs: 
% 
%-------------------------------------------------------------------------% 

clear all; close all; clc;
addpath(genpath('../utils'))

for fig_num = 5
    
    % load pre plot options     
    pre_plot 
    
    for row_num = 1:6
        
        % set example number 
        if row_num <= 3     
            ex = 1;
        else
            ex = 5;
        end
        
   		% set dimension 
        dim = 1;
    
        % set points 
        set_points
        
        for col_num = 1:3
                      
            fig = figure();
 
            if col_num == 1   % plot histogram
                
                % activation and set number
                setup_act_and_set_num
                  
                % load data 
                load_data_histogram
                 
                % save data
                for samp_mode = 1:2
                    x_temp_data   = zeros(max_M_points,dim);
                    x_values_data = [];
                
                    for num_trials = 1:20
                        x_temp_data(:,:) = x_train_save_data(samp_mode,num_trials,:,:);
                        x_values_data    = [x_values_data; x_temp_data];
                    end

                    samples_data(samp_mode,:,:) = x_values_data; 
                end
                 
                for samp_num = 2:-1:1
                    % set names, marker, face, color
                    set_names_marker_face_edge_color_histogram
     
                    x = samples_data(samp_num,:)';  
                    h = histogram(x,'DisplayName',name); 
                    h.Normalization = 'probability';
                    h.EdgeColor = 'none';
                    h.FaceColor = default_color(samp_num,:);
                    h.NumBins = 100;

                    hold on 
                end
                 
                beautify_plot_hist
                  
            elseif col_num == 2 % plot probability distribution 
                
                % activation and set number
                setup_act_and_set_num
                  
                % load data 
                load_data_histogram 
                
                % load training data 
                data_file_name = ['../data/example_' num2str(ex) '_dim_'  num2str(dim) '/'...
                                    'train_data_example_' num2str(ex) '_dim_' num2str(dim)];
                load([data_file_name '.mat']);
                
                % set parameters
                set_param_chris_prob
                
                if Chris_plot
                    Chris_vals(:,num_trials) = Chris_vals_save_data(samp_mode,num_trials,:,nb_training_steps);
                    yplot = Chris_vals;
                else
                    Prob_dist(:,num_trials) = Prob_dist_save_data(samp_mode,num_trials,:,nb_training_steps);
                    yplot = Prob_dist;
                end

                xplot = X(:,1,num_trials);
                
                % sort data
                [xplot, I] = sort(xplot);
                yplot      = yplot(I);
 
                plot(xplot,yplot,...  
                    'LineWidth',2.5,...
                    'Color',default_color(1,:),...
                    'DisplayName',['CAS-' actname]);
                 
                beautify_plot_hist
                hold off 
                
            elseif col_num == 3  % plot function 
                
                % load training data 
                data_file_name = ['../data/example_' num2str(ex) '_dim_'  num2str(dim) '/'...
                                    'train_data_example_' num2str(ex) '_dim_' num2str(dim)];
                load([data_file_name '.mat']);
                 
                xplot = X(:,1,1);
                
                % sort data
                [xplot, I] = sort(xplot);
                yplot      = Y(I);

                plot(xplot,yplot,...
                    'LineWidth',2.5,...
                    'Color',default_color(1,:),...
                    'DisplayName','$f$')
                
                beautify_plot_hist
                
            end
 
  
            namefig    = sprintf('fig_%d_%d_%d',fig_num,row_num,col_num);        
            foldername = ['../figs/Figure' ' ' sprintf('%d',fig_num) '/'];
            saveas(fig, fullfile(foldername, namefig),'epsc');   
            
            hold off
        end
    end
    
   % clear data
   clear all; close all; clc;
end
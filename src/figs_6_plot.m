%-------------------------------------------------------------------------%
% Filename: figs_6_plot.m 
% Author: Juan M. Cardenas, Ben Adcock, Nick Dexter
% Part of the paper "CAS4DL: Christoffel Adaptive Sampling for function 
% approximation via Deep Learning"
%
% Description: generates the plots for Figure 6.
%
% Inputs: 
%
% Outputs: 
% 
%-------------------------------------------------------------------------% 

clear all; close all; clc;
addpath(genpath('../utils'))

for fig_num = 6
    
    % load pre plot options     
    pre_plot 
    
    for row_num = 1:2
        
        % set example number 
        if row_num == 1   
            ex = 1;
        else
            ex = 5;
        end
        
   		% set dimension 
        dim = 1;
    
        % set points 
        set_points
        
        for col_num = 1:3 
            
            % set activation and set number
            set_act_fig6
 
            % load data
            load_data_histogram

            % load training data
            load_training_data 

            % set parameters 
            set_param_basis
              
            Psi_matrix(:,:)   = largest_basis_train_save_data(samp_mode,num_trials,:,:);
            x_train_plot(:,:) = X(:,1,num_trials);
            
            fig = figure();

            for j = 1:nb_basis
                
                xplot = x_train_plot;
                yplot = Psi_matrix(:,j);
                
                % sort data
                [xplot, I] = sort(xplot);
                yplot      = yplot(I);
                
                face_color = default_color(j,:);
                edge_color = default_color(j,:);
                
                plot(xplot,yplot,...
                    'LineWidth',2,...
                    'Color',face_color); 
                hold on
            end

            legend('$\psi_1^{(l_{\mathrm{final}})}$','$\psi_2^{(l_{\mathrm{final}})}$',...
                '$\psi_3^{(l_{\mathrm{final}})}$','$\psi_4^{(l_{\mathrm{final}})}$',...
                '$\psi_5^{(l_{\mathrm{final}})}$','$\psi_6^{(l_{\mathrm{final}})}$',...
                'Interpreter','latex')
         
            beautify_plot_basis 
  
            namefig    = sprintf('fig_%d_%d_%d',fig_num,row_num,col_num);        
            foldername = ['../figs/Figure' ' ' sprintf('%d',fig_num) '/'];
            saveas(fig, fullfile(foldername, namefig),'epsc');   
            
            hold off
        end
    end
    
   % clear data
   clear all; close all; clc;
end
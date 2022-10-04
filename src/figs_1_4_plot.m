%-------------------------------------------------------------------------%
% Filename: figs_1_4_plot.m 
% Author: Juan M. Cardenas, Ben Adcock, Nick Dexter
% Part of the paper "CAS4DL: Christoffel Adaptive Sampling for function 
% approximation via Deep Learning"
%
% Description: generates the plots for Figure 1-4.
%
% Inputs: 
%
% Outputs: 
% 
%-------------------------------------------------------------------------% 

clear all; close all; clc;
addpath(genpath('../utils'))

for fig_num = 4
    
    % load pre plot options     
    pre_plot
    
    % set example number 
    if fig_num == 3
        ex = example_values(end);
    else
        ex = example_values(fig_num);
    end
    
    for row_num = 1:4
        
   		% set dimension 
        if fig_num == 3
            dim = dim_values_f3(row_num);
        else
    		dim = dim_values(row_num);    
        end    
        % set points 
        set_points
        
        for col_num = 1:4
                      
            fig = figure();
 
            for set_num = 1:2

                if set_num == 1 
                    % ReLU and Tanh | 5x50 and 10x100 | normal
                    init_act_num  = 1;      final_act_num  = 2;
                    init_arch_num = 5;      final_arch_num = 6;

                elseif set_num == 2
                    % eLU and sigmoid | 5x50 and 10x100 | normal
                    init_act_num  = 3;      final_act_num  = 3;
                    init_arch_num = 5;      final_arch_num = 6;
                    marker = marker_list(2,:);    

                else
                    disp(['wrong set of experiments'])
                end            
            
                set_act_num = init_act_num:final_act_num;
                            
                % set L2 error or rank     
                if (col_num == 1) || (col_num == 2) 
                    arch_num = init_arch_num;
                else
                    arch_num = final_arch_num; 
                end
                 
                % loop over activation functions
                for act_num = set_act_num  
                    
                    setdir = convertStringsToChars(set_dirnames(arch_num-4));

                    base_dir      = ['cedar_CAS_set' num2str(set_num) '_' setdir '_arch_' set_compar '_' setdate '_'];
                    run_ID        = base_dir; 
                    base_dir_file = ['matlab_' base_dir(1:end-1) '/' run_ID]; 
    
                    
                    actdir  = convertStringsToChars(activations_dirnames(act_num));
                    actname = convertStringsToChars(activations(act_num));
                    
                    arch_name = [num2str(arch_layers(arch_num)) 'x' num2str(arch_nodes(arch_num))];
                    
                    filename  = ['../data/' base_dir_file actdir '_default_' num2str(arch_layers(arch_num))...
                                'x' num2str(arch_nodes(arch_num)) '_' num2str(points,'%06.f')...
                                '_pnts_5e-07_tol_Adam_opt_example_' num2str(ex) '_dim_' num2str(dim)];
                    
                    disp(filename)
                    load([filename '_extracted_data.mat'])

                    % set L2 error or rank     
                    if (col_num == 1) || (col_num == 3)
                        y_values_mean_data = squeeze(mean(L2_error_save_data,2));
                    else
                        y_values_mean_data = squeeze(mean(r_values_save_data,2));
                    end
                    
                    x_values_mean_data = squeeze(mean(M_values_save_data,2)); 
                                         
                    % L2 error data vs M values 
                    for samp_num = 1:2     
                        
                        % set names, maker, face, edge, color
                        set_names_marker_face_edge_color
                        
                        hold on 
                    
                        plot(x_values_mean_data(samp_num,:),...
                             y_values_mean_data(samp_num,:),...
                             marker,...
                             'markersize',ms,...
                             'MarkerFaceColor',face_color,...
                             'MarkerEdgeColor',edge_color,...
                             'LineWidth',line_width,...
                             'Color',face_color,...
                             'DisplayName',name);                          
                        
                        data_plot.y_values(count,:)        = y_values_mean_data(samp_num,:);                            
                        data_plot.x_values(count,:)        = x_values_mean_data(samp_num,:);
                        data_plot.marker_list(count,:)     = marker;
                        data_plot.face_color_list(count,:) = face_color;
                        data_plot.edge_color_list(count,:) = edge_color;
                        data_plot.line_width_list(count,:) = line_width;
                        data_plot.name_list{count}         = name;
                             
                        count = count + 1;
                    end     
                end
            end
            
            count = 1;
            
            %legend_from_plot  
            beautify_plot 
            
            namefig    = sprintf('fig_%d_%d_%d',fig_num,row_num,col_num);        
            foldername = ['../figs/Figure' ' ' sprintf('%d',fig_num) '/'];
            saveas(fig, fullfile(foldername, namefig),'epsc');   
            
            hold off
        end
    end
    
   % clear data
   clear all; close all; clc;
end
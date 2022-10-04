%-------------------------------------------------------------------------%
% Filename: figs_8_plot.m 
% Author: Juan M. Cardenas, Ben Adcock, Nick Dexter
% Part of the paper "CAS4DL: Christoffel Adaptive Sampling for function 
% approximation via Deep Learning"
%
% Description: generates the plots for Figure 8.
%
% Inputs: 
%
% Outputs: 
% 
%-------------------------------------------------------------------------% 

clear all; close all; clc
addpath(genpath('../utils'))

dim_vec = [1 2 4 8 16];
plot_H1 = true;
fig_num = 8;

for d = 1:5
    dim_num = dim_vec(d);
    
    load(['../data/PDE data/result_v3_dim_' num2str(dim_num) '.mat'])
    
    x_plot_ASGD = M_values_ASGD;
    x_plot_MC   = M_values_MC; 
    if plot_H1
        y_plot_CAS = avg_H1_ASGD(1:9,:)./H1_u_both;
        y_plot_MC  = avg_H1_MC(1:9,:)./H1_u_both;
    else
        y_plot_CAS = avg_L2_ASGD(1:9,:)./L2_u_both;
        y_plot_MC  = avg_L2_MC(1:9,:)./L2_u_both;
    end
    
    figure(); 
    semilogy(x_plot_ASGD,y_plot_CAS,...
            'DisplayName','5x50 CAS-tanh')
    hold on 
    semilogy(x_plot_MC,y_plot_MC,...
            'DisplayName','5x50 MC-tanh'); 

    legend
    legend_location = 'northeast';
    beautify_plot_PDE
    
    if plot_H1
        ylabel('Average $L^2_\varrho(\Omega;H_0^1(D))$ error','Interpreter','LaTex');
    else
        ylabel('Average $L^2_\varrho(\Omega;L^2(D))$ error','Interpreter','LaTex');
    end
    figHandles = get(groot, 'Children');

    numfigs = size(figHandles,1);

    for j = 1:numfigs
        figHandles(j).CurrentAxes.Position = [0.09 0.12 0.87 0.85];
        figHandles(j).CurrentAxes.InnerPosition = [0.175,0.130,0.80,0.85];
        
        if plot_H1
            V_space = 'H1';
        else
            V_space = 'L2';
        end
        
        %folder_name = 'Figures/'; 
        %namefig    = sprintf('fig_%d_%d_%d',fig_num,row_num,col_num);        
        folder_name = ['../figs/Figure' ' ' sprintf('%d',fig_num) '/'];
        
        print(figHandles(j),[folder_name '/' 'PDE_' V_space '_result_dim_' num2str(dim_num) '.pdf'],'-dpdf'); 
        savefilename = [folder_name '/' 'PDE_' V_space '_result_dim_' num2str(dim_num)];
        saveas(gcf,savefilename,'epsc');
    end 
    
    hold off ; 
end
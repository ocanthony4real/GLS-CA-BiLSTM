%NB: code may take long hours to run using large dataset as such found in
%this file

close all
clc
clear


lb = [0.001 0.005 10];
ub = [0.1 5 150];

step_size = [0.005 0.05 5];
dim = 3;
T = 10;
N = 20;

%%
[Best_score,Best_pos,CO_curve,testY,test_target_grp,mae,mse,smape,r_squared]=CO(step_size, N, T, lb, ub, dim);


pred = testY;

figure, plot(testY)
hold on
plot(test_target_grp, 'r')
title('Prediction Results')
xlabel('Number of Days')
ylabel('Power');
legend('GLS-CA-BiLSTM','Actual Demand')

disp(['Mean Absolute Error (MAE): ', num2str(mae)]);
disp(['Mean Squared Error (MSE): ', num2str(mse)]);
disp(['Symmetric Mean Absolute Percentage Error (SMAPE): ', num2str(smape)]);
disp(['Coefficient of Determination (R^2): ', num2str(r_squared)]);

disp(['Minimum objective function value is ', num2str(Best_score)]);
disp(['Optimal Learning rate, Gradient threshold, and number of layers are ', num2str(Best_pos)]);
%%



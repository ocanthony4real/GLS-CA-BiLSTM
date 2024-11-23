
function [test_target_grp, testY, MAE] = myCostFunction(LearnRate, GradThresh, Layer1)
global length_test_target
%%
file = 'UDU_turb_powers.xlsx'; %insert data file path
data = readtable(file, 'Sheet', 1, 'PreserveVariableNames', true, 'ReadVariableNames', true);
data_array = table2array(data);

max_vals = max(data_array, [], 1);  % max_vals stores the max of each column
for i = 1:size(data_array, 2)
    data_array(:, i) = data_array(:, i) / max_vals(i);  % Normalize each column
end


%% Split into features and target
features = data_array(:, 1:end-1);
target = data_array(:, end);
num_rows = size(features, 1);
train_rows = num_rows - 371;
train_features = features(1:train_rows, :);
train_target = target(1:train_rows, :);
test_features = features(train_rows+1:end, :);
test_target = target(train_rows+1:end, :);
length_test_target = size(test_target,1);
%% rearrange train data
i=1;
while ~isempty(train_features)
    pick=7;
    if pick<size(train_features,1)
        train_feat_grp{i}=(train_features(1:pick,:))';
        train_target_grp(i)=train_target(pick);
        train_features(1,:)=[];
        train_target(1,:)=[];
        i=i+1;
    else
        train_feat_grp{i}=train_features';
        train_target_grp(i)=train_target(end);
        break;
        
    end
end

%% rearrange test data
i=1;
while ~isempty(test_features)
    pick=7;
    if pick<size(test_features,1)
        test_feat_grp{i}=(test_features(1:pick,:))';
        test_target_grp(i)=test_target(pick);
        test_features(1,:)=[];
        test_target(1,:)=[];
        i=i+1;
    else
        test_feat_grp{i}=test_features';
        test_target_grp(i)=test_target(end);
        break;
        
    end
end


 Layer1 = ceil(Layer1);
maxEpochs = 70;
inputSize = 5;
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(Layer1, 'OutputMode', 'last')       
    fullyConnectedLayer(100)                    
    reluLayer
    fullyConnectedLayer(50)                     
    fullyConnectedLayer(1)                      
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ... 
    'InitialLearnRate',LearnRate, ...
    'GradientThreshold',GradThresh, ...
    'Shuffle','never', ...
    'Verbose',0);
net = trainNetwork(train_feat_grp, train_target_grp',layers,options);
% save('net.mat','net')
testY = double(predict(net,test_feat_grp));
test_target_grp = test_target_grp * max_vals(6);
testY = testY * max_vals(6);



MAE = mean(abs(test_target_grp - testY'));  

end

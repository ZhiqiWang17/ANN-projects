%% assignment
% Compare train algorithms: traingd, traingda, traingdx, traincgf, trainbfg, trainlm

clear
clc
close all

% Configuration:
alg1 = 'traingd'; % gradient descent
alg2 = 'traingda';% gradient descent with adaptive learning rate
alg3 = 'traingdx'; % gradient descent with momentum and adaptive learning rate
alg4 = 'trainbfg'; % BFGS quasi Newton algorithm
alg5 = 'trainlm'; % Levenberg-Marquardt algorithm
alg6 = 'trainbr'; % Bayesian regularisation
H = 30;% Number of neurons in the hidden layer, also change to 5, 20

sigma=0.1;

%generation of examples and targets
x=0:0.05:3*pi; y=sin(x.^2)+sigma*randn(1,length(x));
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%creation of networks
netNum=6;
nets{1}=feedforwardnet(H,alg1);
nets{2}=feedforwardnet(H,alg2);
nets{3}=feedforwardnet(H,alg3);
nets{4}=feedforwardnet(H,alg4);
nets{5}=feedforwardnet(H,alg5);
nets{6}=feedforwardnet(H,alg6);

% Calculate the training time for each algorithm
training_time = zeros(1, netNum);

% Calculate the convergence speed for each algorithm
convergence_speed = zeros(1, netNum);

for i=1:netNum
    nets{i}.divideParam.trainRatio = 70/100; % Training set percentage
    nets{i}.divideParam.valRatio = 15/100; % Validation set percentage
    nets{i}.divideParam.testRatio = 15/100; % Test set percentage
    nets{i}.trainParam.epochs=1000;
    nets{i}.trainParam.max_fail = 6; % Set maximum validation failures to enable early stopping
    
    tic;
    [nets{i},tr{i}]=train(nets{i},p,t);
    training_time(i) = toc;
    
    simulation{i}=sim(nets{i},p);
end

final_MSE = zeros(1, netNum);
for i = 1:netNum
    final_MSE(i) = tr{i}.perf(end);
end

disp('Final MSE value for each algorithm:');
disp(array2table(final_MSE, 'VariableNames', {alg1, alg2, alg3, alg4, alg5, alg6}));
% Display the training time for each algorithm
disp('Training time (seconds) for each algorithm:');
disp(array2table(training_time, 'VariableNames', {alg1, alg2, alg3, alg4, alg5, alg6}));

% Plot the Mean Square Error for each algorithm on test set
figure;
hold on;

for i=1:netNum
    plot(tr{i}.perf,'DisplayName',nets{i}.trainFcn,'LineWidth',1.5);
end
set(gca, 'YScale', 'log')
xlabel('Epoch');
ylabel('Mean Square Error');
title('MSE over epocs');
legend ('show');
hold off;


% Plot the regression results 
figure;
for i=1:netNum
    subplot(2, 3, i);
    postregm(cell2mat(simulation{i}), y);
    title(nets{i}.trainFcn);
end


% Define the colormap colors for each algorithm
colors = lines(netNum);

% Store the epochs you want to check in a vector
epochs_to_check = [3, 30, 100];

% Train the network for 100 epochs
for i = 1:netNum
    nets{i}.trainParam.epochs = 100;
    [nets{i}, tr{i}] = train(nets{i}, p, t);
    simulation{i} = sim(nets{i}, p);
end

% Create a new figure to plot the fitted curves for all algorithms
figure;
hold on;

% Plot the fitted curve for each algorithm in a different color
for i = 1:netNum
    plot(x, cell2mat(simulation{i}(end,:)), 'LineWidth', 1.5, 'Color', colors(i, :), 'DisplayName', nets{i}.trainFcn);
end

% Plot the actual data points
plot(x, y, 'k *');

% Add axis labels and a legend
xlabel('x');
ylabel('y');
title('Fitted Curves at 100th Epoch');
% Get the current legend labels
leg = legend();

% Modify the label for the target data
leg.String{7} = 'target';

hold on;

% Create a figure of regression in different epocs
figure;

% Iterate over the epochs
for e = 1:length(epochs_to_check)
    epoch_to_check = epochs_to_check(e);
    R_values = zeros(1, netNum);

    for i = 1:netNum
        % Create a new network for each epoch
        temp_net = feedforwardnet(H, nets{i}.trainFcn);
        temp_net.trainParam.epochs = epoch_to_check;
        
        % Train the network for epoch_to_check epochs
        temp_net = train(temp_net, p, t);

        % Simulate the trained network
        temp_sim = sim(temp_net, p);

        % Calculate the R value
        R = corrcoef(cell2mat(temp_sim), y);
        R_values(i) = R(1, 2);
    end

    % Create a subplot for each epoch
    sp = subplot('Position', [(0.33 * (e - 1) + 0.06) 0.15 0.26 0.7]);


    % Plot the R values for each algorithm at the specified epoch
    for i = 1:netNum
        bar(i, R_values(i), 'FaceColor', colors(i, :));
        hold on;
    end
    hold off;
    xticks(1:netNum);
    xticklabels({nets{1}.trainFcn, nets{2}.trainFcn, nets{3}.trainFcn, nets{4}.trainFcn, nets{5}.trainFcn, nets{6}.trainFcn});
    xlabel('Algorithm');
    ylabel(['R Value']);
    title(['Regression Value at Epoch ' num2str(epoch_to_check)]);
end


%%

% Define the data
alg_names = {'traingd', 'traingda', 'traingdx', 'trainbfg', 'trainlm', 'trainbr'};
R_orig = [0.556, 0.778, 0.871, 0.987, 1, 1];
MSE_orig = [0.316, 0.221, 0.189, 0.019, 6.0658e-05, 1.5967e-04];
Train_time_orig = [1.9819, 1.5621, 1.591, 5.4039, 3.7402, 4.9967];
R_noisy = [0.505, 0.695, 0.803, 0.971, 0.96, 0.993];
MSE_noisy = [0.36644, 0.25679, 0.17435, 0.028497, 0.038861, 0.0068777];
Train_time_noisy = [1.9017, 1.494, 1.4356, 4.5015, 3.8147, 4.6622];

% Define the colormap colors for each algorithm
colors = [    0.9290 0.6940 0.1250; % yellow    
            0.4940 0.1840 0.5560; % purple    
            0.4660 0.6740 0.1880; % green    
            0.3010 0.7450 0.9330; % light blue    
            0.6350 0.0780 0.1840; % dark red    
            0.9290 0.3290 0.1370]; % orange

% Plot the R values for original and noisy data
figure;
subplot(1,3,1);
h = bar([R_orig; R_noisy]', 'grouped');
for i = 1:size(h,2)
    h(i).FaceColor = colors(i,:);
end
xticklabels(alg_names);
xlabel('Algorithm');
ylabel('R Value');
title('Regression Value Comparison');
legend('Original', 'Noisy');

% Plot the MSE values for original and noisy data
subplot(1,3,2);
h = bar([MSE_orig; MSE_noisy]', 'grouped');
for i = 1:size(h,2)
    h(i).FaceColor = colors(i,:);
end
xticklabels(alg_names);
xlabel('Algorithm');
ylabel('MSE');
title('MSE Comparison');
legend('Original', 'Noisy');

% Plot the training time for original and noisy data
subplot(1,3,3);
h = bar([Train_time_orig; Train_time_noisy]', 'grouped');
for i = 1:size(h,2)
    h(i).FaceColor = colors(i,:);
end
xticklabels(alg_names);
xlabel('Algorithm');
ylabel('Training Time (s)');
title('Training Time Comparison');
legend('Original', 'Noisy');

% Add hatching to the bars
for i = 1:size(h,2)
    hatchfill2(h(i), 'single', 45, 5, 'black', 'none', 1);
end


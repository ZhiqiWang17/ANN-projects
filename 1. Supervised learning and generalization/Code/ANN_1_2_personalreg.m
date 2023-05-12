%student number: r0822618
d1=8;
d2=8;
d3=6;
d4=2;
d5=2;


% load and define datasets
load("Data_Problem1_regression.mat");
Tnew = (d1*T1 + d2*T3 + d3*T3 + d4*T4 + d5*T5)/(d1+d2+d3+d4+d5);

%Training set
temp = datasample([X1 X2 Tnew],1000,1);
trainingX = temp(:,1:2).';
trainingY = temp(:,3).';
trainingP = con2seq(trainingX);
trainingT = con2seq(trainingY);
%Validation set
temp = datasample([X1 X2 Tnew],1000,1);
validationX = temp(:,1:2).';
validationY = temp(:,3).';
validationP = con2seq(validationX);
validationT = con2seq(validationY);
%Test set
temp = datasample([X1 X2 Tnew],1000,1);
testX = temp(:,1:2).';
testY = temp(:,3).';
testP = con2seq(testX);
testT = con2seq(testY);

%Plot training set surface
x = trainingX(1,:).';
y = trainingX(2,:).';
xlin = linspace(min(x),max(x),33);
ylin = linspace(min(y),max(y),33);
[X,Y] = meshgrid(xlin,ylin);
f = scatteredInterpolant(x,y,trainingY.');
Z = f(X,Y);
figure
mesh(X,Y,Z) %interpolated
axis tight; hold on
plot3(X,Y,Z,'.','MarkerSize',10) %nonuniform


% build the ff network
hiddenLayerSizes = 20;
net = feedforwardnet(hiddenLayerSizes);
net.trainFcn = 'trainlm';
net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
net.trainParam.epochs = 1000;
[trainedNet, tr] = train(net, trainingP, trainingT);

postregm(cell2mat(sim(trainedNet, trainingP)), cell2mat(trainingT));
postregm(cell2mat(sim(trainedNet, validationP)), cell2mat(validationT));

%Test set error
mseTest = mean((testY-cell2mat(sim(trainedNet,testP))).^2);

% 1. Simulate the network's output for the test set
testOutput = sim(trainedNet, testP);

% 2. Plot the surface of the test set and the approximation given by the network
figure;
testOutputSurface = scatteredInterpolant(testX(1,:).', testX(2,:).', cell2mat(testOutput).');
Z_testOutput = testOutputSurface(X, Y);
mesh(X, Y, Z_testOutput);
hold on;
plot3(testX(1,:), testX(2,:), cell2mat(testOutput), '.', 'MarkerSize', 10);
xlabel('X1');
ylabel('X2');
zlabel('Output');
title('Test Set Surface and Network Approximation');

% 3. Plot the error level curves
figure;
error = testY - cell2mat(testOutput);
contourf(X, Y, Z - Z_testOutput);
colorbar;
xlabel('X1');
ylabel('X2');
title('Error Level Curves - trainbr');

% 4. Compute the Mean Squared Error (MSE) on the test set
fprintf('Mean Squared Error on the test set: %.8f\n', mseTest);



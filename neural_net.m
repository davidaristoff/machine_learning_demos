%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Self-contained feed-forward neural network algorithm for regression.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Includes demo of fitting data that follows a nonlinear curve.
%Running the neural net requires ~50 lines of low-level code; 
%in particular, this code does NOT call any high-level functions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define underlying nonlinear function 
G = @(x) cos(3*pi*x) - x + 1;

%define training data set
N = 100; xhats = rand(1,N)-0.5; yhats = G(xhats) + 0.1*normrnd(0,1,[1 N]);

%define net architecture, input & output dimension, and learning rate
layers = 3; layer_width = 10; d_in = 1; d_out = 1; alph = 0.2;

%define activation function and its derivative
f = @(x) 1./(1 + exp(-x)); df = @(x) exp(-x)./(exp(-x) + 1).^2;

%initialize neural net
params = initialize(layers,layer_width,d_in,d_out,'r');

%initialize figure
close all; figure('DefaultAxesFontSize',18,'Position',[100 100 1100 700])
plot(xhats,yhats,'.b','markersize',20); hold on
plot([-.5 .5],[-1 -1],'-r','linewidth',2);
legend('training data','neural net fit','interpreter','latex', ...
    'fontsize',24,'autoupdate','off'); h = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%train neural network

%define total # of gradient descent steps, and # of samples per batch
steps = 5*10^2; lagtime = 100; samples = 10;

tic

%begin training
for step = 1:steps
    for time = 1:lagtime

    %initialize gradients of parameters
    batch_grads = initialize(layers,layer_width,d_in,d_out,'z');

    %compute sum of minibatch gradients
    for sample = 1:samples

        %choose random sample
        i = randi([1 N]); xhat = xhats(i); yhat = yhats(i);
    
        %run forward pass
        inputs = forward_pass(xhat,params,f,layers);
    
        %run back propagation
        grads = back_propagate(yhat,inputs,params,f,df,layers);
    
        %update gradients of batch
        for layer=1:layers
            batch_grads{layer} = batch_grads{layer} + grads{layer};
        end

    end

    %update parameters using minibatch gradients
    for layer=1:layers
        params{layer} = params{layer} - (alph/samples)*batch_grads{layer};
        %Euler's method, Delta t = (alph/samples), -batch_grads =
        %derivative
    end

    end

    h = plot_net(params,f,layers,h);

end
%end training

toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot neural network points

function h = plot_net(params,f,layers,h)

xvals = -0.5:0.02:0.5; L = length(xvals); yvals = zeros(1,L);
for n=1:L
    x = xvals(n); 
    inputs = forward_pass(x,params,f,layers); 
    yvals(n) = inputs{layers+1};
end
delete(h); h = plot(xvals,yvals,'-r','linewidth',2); 
axis([-.5 .5 -.5 2.5]); pause(0.001); 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize neural net parameters

function params = initialize(layers,layer_width,d_in,d_out,type)

if type == 'r'
    params = cell(layers,1);
    params{1} = [rand(layer_width,d_in),zeros(layer_width,1)]; 
    params{layers} = [rand(d_out,layer_width),zeros(d_out,1)];
    params(2:layers-1) = ...
        {[rand(layer_width,layer_width),zeros(layer_width,1)]};
else
    params = cell(layers,1);
    params{1} = zeros(layer_width,d_in+1); 
    params{layers} = zeros(d_out,layer_width+1);
    params(2:layers-1) = {zeros(layer_width,layer_width+1)};
end

end

%type 'r' is random initialization; type 'z' is all zeros initialization
%each cell in params gives the parameters associated to one layer
%the last column of a cell is the bias; the other columns are the weights

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%forward pass function
%computes input to each NN layer; last input is the output layer

function inputs = forward_pass(xhat,params,f,layers)

%initialize layers inputs and outputs
inputs = cell(layers+1,1);

%pull and store initial input
out = xhat; inputs{1} = xhat; 

for layer = 1:layers
    
    %pull weights and bias at current layer
    W = params{layer}(:,1:end-1); b = params{layer}(:,end);
    
    %compute and store input to current layer, and compute output
    in = W*out + b; inputs{layer+1} = in; out = f(in);
    
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%back propagation function
%computes gradients of parameters with respect to mean square loss function

function grad_params = back_propagate(yhat,inputs,params,f,df,layers)

%initialize gradients, pull net output, and define initial errors
grad_params = params; y = inputs{layers+1}; errors = y-yhat;

for layer = layers:-1:1

    %pull weights and bias at current layer, and layer inputs
    W = params{layer}(:,1:end-1); in = inputs{layer};

    %compute gradients of weights
    grad_params{layer}(:,1:end-1) = errors*f(in)';

    %compute gradients of biases
    grad_params{layer}(:,end) = errors;

    %update gradient of loss with respect to current layer input
    errors = df(in).*(W'*errors);
    
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

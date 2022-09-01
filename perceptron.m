%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Demo of a perceptron (1 artificial neuron) for fitting binary class data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define point, p, on dividing surface, along with parallel vector, v
p = [1;2]; v = [1;1];

%generate data from class 0 and class 1
N = 500;
s = normrnd(0,1,[N 1]); t = rand(N,1);
C0 = [v(1)*s,v(2)*s] - [-v(2),v(1)].*normrnd(1,.8,[N,1]);
C1 = [v(1)*s,v(2)*s] + [-v(2),v(1)].*normrnd(1,.8,[N,1]);

%define activation function and its derivative
f = @(x) 1./(1 + exp(-x)); df = @(x) exp(-x)./(exp(-x) + 1).^2;

%define initial parameter values and mesh in parameter space
w1 = 0; w2 = -1; b = -.5; w1_ = -2; w2_ = 2; b_ = 0.5; s = 12;
[W1,W2,B] = meshgrid(w1_:(w1-w1_)/s:w1,w2:(w2_-w2)/s:w2_,b:(b_-b)/s:b_);

%compute direction field for negative gradient of loss function
U = zeros(size(B)); V = zeros(size(B)); W = zeros(size(B));
for n=1:N   %compute gradients over class 0
    x1 = C0(n,1); x2 = C0(n,2); in = W1*x1 + W2*x2 + B;
    U = U - f(in).*df(in)*x1; 
    V = V - f(in).*df(in)*x2;
    W = W - f(in).*df(in);
end
for n=1:N   %compute gradients over class 1
    x1 = C1(n,1); x2 = C1(n,2); in = W1*x1 + W2*x2 + B;
    U = U - (f(in)-1).*df(in)*x1; 
    V = V - (f(in)-1).*df(in)*x2;
    W = W - (f(in)-1).*df(in);
end

%plot data, direction field, and parameter vector
xs = -5:.1:5; close all; h = [];
figure('DefaultAxesFontSize',18,'Position',[100 100 1500 700]); 
pl1 = subplot(1,2,1);
h(1) = scatter(C0(:,1),C0(:,2),20,'b','filled'); hold on
h(2) = scatter(C1(:,1),C1(:,2),20,'r','filled');
lp = plot(xs,-(w1/w2)*xs-b/w2,'-.g','linewidth',2);  axis([-5 5 -5 5]);
xlabel('$x_1$','interpreter','latex'); 
ylabel('$x_2$','interpreter','latex');
legend({'class 0','class 1', ...
    '$w_1 x_1 + w_2 x_2 + b = 0$'},'interpreter','latex', ...
    'fontsize',18,'AutoUpdate','off')
title('decision boundary','interpreter','latex')
pl2 = subplot(1,2,2); quiver3(W1,W2,B,U,V,W); hold on;
pp = plot3(pl2,w1,w2,b,'.r','markersize',20);
axis([w1_ w1 w2 w2_ b b_])
xlabel('$w_1$','interpreter','latex'); 
ylabel('$w_2$','interpreter','latex');
zlabel('$b$','interpreter','latex');
title('parameter values','interpreter','latex');

%train the perceptron by following direction field in parameter space
steps = 2*10^2; dt = 0.1; 
for step = 1:steps
    grad = [0;0;0];
    for n=1:N   %compute gradients over class C0
        x1 = C0(n,1); x2 = C0(n,2); in = w1*x1 + w2*x2 + b;
        grad = grad - [f(in)*df(in)*x1;f(in)*df(in)*x2;f(in)*df(in)];
    end
    for n=1:N   %compute gradients over class C1
        x1 = C1(n,1); x2 = C1(n,2); in = w1*x1 + w2*x2 + b;
        grad = grad - ...
            [(f(in)-1)*df(in)*x1;(f(in)-1)*df(in)*x2;(f(in)-1)*df(in)];
    end
    w1 = w1+(dt/N)*grad(1); w2 = w2+(dt/N)*grad(2); b = b+(dt/N)*grad(3);
    pause(0.0001); delete(pp); delete(lp);
    pp = plot3(pl2,w1,w2,b,'.r','markersize',20);
    lp = plot(pl1,xs,-(w1/w2)*xs-b/w2,'-.g','linewidth',2); 
    axis(pl1,[-5 5 -5 5]); 
end
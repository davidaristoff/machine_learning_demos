%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Self-contained kernel ridge regression algorithm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Includes demo of fitting data that follows a nonlinear curve.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define underlying nonlinear function 
G = @(x) cos(3*pi*x) - x + 1;

%define training data set
N = 100; xs = rand(N,1)-0.5; ys = G(xs) + 0.1*normrnd(0,1,[N 1]);

%define RBF kernel
h = 0.1; K = @(x,y) exp(-(x-y).^2/h);

%define kernel matrix, identity, and regularization parameter
M = zeros(N,N);
for i=1:N
    for j=1:N
        M(i,j) = K(xs(i),xs(j));
    end
end
I = eye(N); lambda = 1;

%solve linear system for coefficients
alph = (M+lambda*I)\ys;

%define interpolation function
f = @(x) sum(alph.*K(x,xs));

%plot interpolation function and data points
plot(xs,ys,'.b','markersize',20); hold on;
plot(-.5:.01:.5,f(-.5:.01:.5),'-r','linewidth',2)
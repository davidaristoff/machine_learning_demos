%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% simulation of a perceptron dividing binary data into two classes  %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%define class 0 (blue) and class 1 (red) data points
C0 = [-1 -1 0 1 1 2 1 4; 2 -1 0 1 -1 0 4 2];   %class 0 (blue o's)
C1 = [0 0 1 2 2 2 3 3; 3 1 5 2 3 4 1 5];       %class 1 (red x's)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%define activation function and loss function

f = @(x) 1./(1+exp(-x));             %sigmoid activation function 
df = @(x) exp(-x)./(exp(-x)+1).^2;   %derivative of activation function

P0 = @(m,b) f(C0(2,:)-m.*C0(1,:)-b);     %model-predicted values on class 0
dP0 = @(m,b) df(C0(2,:)-m.*C0(1,:)-b);   %derivatives
P1 = @(m,b) f(C1(2,:)-m.*C1(1,:)-b);     %model-predicted values on class 1
dP1 = @(m,b) df(C1(2,:)-m.*C1(1,:)-b);   %derivatives

L = @(m,b) 0.5*sum((P0(m,b)-0).^2+(P1(m,b)-1).^2);    %loss function
dLm = @(m,b) sum(-C0(1,:).*(P0(m,b)-0).*dP0(m,b)...   %derivative w.r.t. m
                 -C1(1,:).*(P1(m,b)-1).*dP1(m,b));
dLb = @(m,b) sum(-(P0(m,b)-0).*dP0(m,b)...            %derivative w.r.t. b
                 -(P1(m,b)-1).*dP1(m,b));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot data points and contours of loss function

m = -2; b = -3.5;   %initial slope and intercept parameters
xs = [-2 5];        %range of x values for plotting

%initialize plot for animation
[fig1,fig2,boundary,params] = initialize_plot(m,b,L,C0,C1,xs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%train the perceptron by steepest descent (AKA gradient descent)

dt = 0.02;   %time step for Euler's method

for t=1:2000
    %do one step of Euler's method
    m = m - dt*dLm(m,b);
    b = b - dt*dLb(m,b);

    %update animation
    [fig1,fig2,boundary,params] = ...
        update_plot(m,b,fig1,fig2,boundary,params,xs);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The above is the essential code; below are just some plotting functions %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize plots of contours of loss function and initial parameters

function [fig1,fig2,boundary,params] = initialize_plot(m,b,L,C0,C1,xs)

    %get data for contour plot
    [X,Y] = meshgrid(-4:0.1:4);
    for i=1:length(X)
        for j=1:length(X)
            Z(i,j) = L(X(1,j),X(1,i));
        end
    end
    
    %open a new figure
    close all; set(0,'defaultTextInterpreter','latex');
    %figure('DefaultAxesFontSize',18,'Position',[100 100 1150 400]);
    figure('DefaultAxesFontSize',18,'Position',[100 100 1400 500]);

    %create first figure of data points
    fig1 = subplot(1,2,1); 
    plot(C0(1,:),C0(2,:),'ob','linewidth',2,'markersize',10); hold on;
    plot(C1(1,:),C1(2,:),'xr','linewidth',2,'markersize',10); 
    axis([-2 5 -2 6]);
    xlabel('$x$'); ylabel('$y$'); title('binary classifier')

    %create second figure of loss function contours
    fig2 = subplot(1,2,2); 
    contourf(X,Y,Z,30); colorbar; hold on;
    xlabel('$m$'); ylabel('$b$'); title('Loss function')
    
    %define initial parameters and decision boundary
    boundary = plot(fig1,xs,m*xs+b,'-g','linewidth',2);
    legend(fig1,'class 0','class 1','$y=mx+b$', ...
        'interpreter','latex','fontsize',18,'autoupdate','off', ...
        'location','southeast')
    params = plot(fig2,m,b,'.g','markersize',30); pause(3)

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%update plots of parameters and decision boundary 

function [fig1,fig2,boundary,params] = ...
    update_plot(m,b,fig1,fig2,boundary,params,xs);

    %delete current decision boundary and parameter point
    delete(boundary); delete(params);
    
    %plot data points and decision boundary
    boundary = plot(fig1,xs,m*xs+b,'-g','linewidth',2);

    %plot current m and b on top of loss function
    params = plot(fig2,m,b,'.g','markersize',30);

    %pause briefly
    pause(0.001)

end
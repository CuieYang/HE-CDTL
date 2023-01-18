% Please write your name and student number below
%
% Name: Jacob Gavin, Laura Pijnacker
% Student number: s4658981, s4332199
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Weights that needs to be initialized
function [f_a_3, E_W, W_1, W_2, result] = mainMLP(iteration_number, learning_rate, hiddenUnits, ytrain, Xtest, Xtraining, ytest)
    % Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %
    % We have to transpose ytrain since it should be on the other side

    Y_trainNEW = transpose(ytrain);
    X_trainingNEW = transpose(Xtraining);
    X_testNEW = transpose(Xtest);
    
    face = reshape(X_testNEW(:,999),48,48);
    imagesc(face);
    colormap(gray);
    for iteration = 1 : iteration_number %
        % %
         disp(['Iteration: ' num2str(iteration) ' / ' num2str(iteration_number)]); %
        % %
         [E_W, grad_E_W_1, grad_E_W_2] = error(Y_trainNEW, W_1, W_2, X_trainingNEW); %
        % %
        disp(['Error: ' num2str(E_W)]); %
        % After each traning iteration we update the weights based on learning
        % rate and the gradient of the error function
         W_1 = W_1 - learning_rate * grad_E_W_1; %
         W_2 = W_2 - learning_rate * grad_E_W_2; %
        % %
    end 
    % %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %
    [~, f_a_3] = forwardprop(W_1, W_2, X_testNEW); %
    % %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %output = round(f_a_3).'
    %result = numel(find(output~=ytest))
    %tolerance = 0.3;
    %result = abs(ytest-f_a_3.') <= tolerance;
    %correct = 0;
    %false = 0;
    %for x=1:size(result, 1)
    %    result(x)
    %    if result(x) == ytest(x)
    %        correct = correct + 1;
    %    end
    %    if result(x) ~= ytest(x)
    %        false = false + 1;
    %    end
    %end
    %correct
    %false
    %total = size(result, 1)
    
    results = round(f_a_3.');
    totalerror = numel(find(results~=ytest))
        
    %representations = col2im(f_a_3(:,1) , [1 1000], [1000 1000], 'distinct');
    %imagesc(representations);
    %colormap(gray);
    
    
end
% You should now implement the sigmoid function. Your implementation should
% compute and return the output of the sigmoid function and its gradient.

function [f_a, grad_f_a] = sigmoid(a)
    % First, compute the output of the sigmoid function (i.e. f_a). Write your
    % code below:
    % -------------------------------------------------------------------------
     f_a = 1 ./ (1+exp(-a));
    % -------------------------------------------------------------------------
    % Once you have computed f_a, use it to compute the gradient of the sigmoid
    % function (i.e. grad_f_a). Write your code below:
    % -------------------------------------------------------------------------
     grad_f_a = f_a .* (1-f_a);
    %grad_f_a = ((exp(-a))/((1+exp(-a)).^2));
    % -------------------------------------------------------------------------
end

% You should now implement the forward propagation function. Your
% implementation should compute and return the outputs of the second and
% third layer units as well as their gradients.

function [f_a_2, f_a_3, grad_f_a_2, grad_f_a_3] = forwardprop(W_1, W_2, X)
    % First, compute the inputs of the second layer units (i.e. a_2). Write
    % your code below:
    % -------------------------------------------------------------------------
    
    a_2 = W_1 * X;  % where X is Xtrain and W_1 are weights of first layer
    % -------------------------------------------------------------------------
    % Once you have computed a_2, use it with the sigmoid function that you
    % have implemented (i.e. sigmoid) to compute the outputs of the second
    % layer units (i.e. f_a_2) and their gradients (i.e. grad_f_a_2). Write
    % your code below:
    % -------------------------------------------------------------------------

    [f_a_2, grad_f_a_2] = sigmoid(a_2);

    % -------------------------------------------------------------------------
    % Then, compute the inputs of the third layer units (i.e. a_3). Write your
    % code below:
    % -------------------------------------------------------------------------
    a_3 = W_2 * f_a_2;
    % -------------------------------------------------------------------------
    % Once you have computed a_3, use it with the sigmoid function that you
    % have implemented (i.e. sigmoid) to compute the outputs of the third layer
    % units (i.e. f_a_3) and their gradients (i.e. grad_f_a_3). Write your code
    % below:
    % -------------------------------------------------------------------------

     [f_a_3, grad_f_a_3] = sigmoid(a_3);

    % -------------------------------------------------------------------------
end
% You should now implement the back propagation function. Your
% implementation should compute and return the gradients of the error
% function.
function [grad_E_W_1, grad_E_W_2] = backprop(f_a_2, f_a_3, grad_f_a_2, grad_f_a_3, T, W_2, X)
    % First, compute the errors of the second and third layer units (i.e.
    % delta_2 and delta_3). Write you code below:
    % -------------------------------------------------------------------------
    delta_3 = grad_f_a_3 .* (f_a_3 - T);
    delta_2 = grad_f_a_2 .* (W_2.' * delta_3);
     % -------------------------------------------------------------------------
    % Once you have computed delta_2 and delta_3, use them to compute the
    % gradients of the error function (i.e. grad_E_W_1 and grad_E_W_2). Write
    % your code below:
    % -------------------------------------------------------------------------
    %[~, grad_E_W_1] = sigmoid(delta_3);
    %[~, grad_E_W_2] = sigmoid(delta_2);
   
    grad_E_W_1 = delta_2 * X';
    grad_E_W_2 = delta_3 * f_a_2';
    % -------------------------------------------------------------------------
    
end
% You should now implement the error function. Your implementation should
% compute and return the output of the error function and its gradient.

function [E_W, grad_E_W_1, grad_E_W_2] = error(T, W_1, W_2, X)
    % First, use the forward propagation function that you have implemented
    % (i.e. forwardprop) to compute the outputs of the second and third layer
    % units (i.e. f_a_2 and f_a_3) as well as their gradients (i.e. grad_f_a_2
    % and grad_f_a_3). Write your code below:
    % -------------------------------------------------------------------------
    [f_a_2, f_a_3, grad_f_a_2, grad_f_a_3] = forwardprop(W_1, W_2, X);
    % -------------------------------------------------------------------------
    % Once you have computed f_a_3, use it to compute the output of the error
    % function (i.e. E_W). Write your code below:
    % -------------------------------------------------------------------------
   
     E_W = 0.5 * sum(sum((f_a_3-T) .^2));
    % -------------------------------------------------------------------------
    % Once you have computed f_a_2, f_a_3, grad_f_a_2 and grad_f_a_3, use them
    % with the back propagation function that you have implemented (i.e.
    % backprop) to compute the gradients of the error function (i.e. grad_E_W_1
    % and grad_E_W_2). Write your code below:
    % -------------------------------------------------------------------------
    [grad_E_W_1, grad_E_W_2] = backprop(f_a_2, f_a_3, grad_f_a_2, grad_f_a_3, T, W_2, X);
    % -------------------------------------------------------------------------
end
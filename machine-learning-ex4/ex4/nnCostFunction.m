function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X=[ones(m,1) X];%adding column of 1s to X

 
a1=X;
z2=Theta1*a1';%25*5000matrix
a2=sigmoid(z2);
a2=[ones(1,size(a2,2)) ; a2];%adding row of ones,now 26*5000matrix
z3=Theta2*a2;
a3=sigmoid(z3);%10*5000 matrix where each column is a training example and each row is h_thetax1,h_thetax2...


theta_xi=a3;

for i=1:size(y,1),%1:5000

y1=zeros(1,num_labels);%1*10 row vector
y1(y(i))=1;
y1=y1'; %10*1 column vector 
coli=theta_xi(:,i);%i th column of matrix a3
part1= -y1.*log(coli);
part2= (1-y1).*log(1-coli);
ans= part1-part2;
theta_xi(:,i)=ans;
end;

%now theta_xi is a 10*5000 matrix where each column is another example and each row is cost of neuron 1,2..

unregJ=1/m*( sum(sum(theta_xi)) );

%adding regularization
sumtheta1=sum(sum( (Theta1(:,2:end)).^2 ));
sumtheta2=sum(sum( (Theta2(:,2:end)).^2 ));

J= unregJ + (lambda/(2*m))*(sumtheta1+sumtheta2);


%backpropagation..

capDelta1=zeros(size(Theta1));%capDelta=capital Delta in layer 1 i.e the triangle symbol ,25*401 matrix
capDelta2=zeros(size(Theta2));% 10*26 mtarix

for t= 1:m,

a_1=X(t,:);%1*401 row vector
%column of ones has already been added to X before
z_2=Theta1*a_1';%25*1 matrix containing layer2 a1,a2.. for 25 neurons
a_2=sigmoid(z_2);
a_2=[1;a_2]; %26*1 matrix
z_3=Theta2*a_2; %10*1 matrix , each row =h_theta1,h_theta2..
a_3=sigmoid(z_3);

y_t=zeros(num_labels,1);%10*1 column vector
y_t(y(t))=1;
del_3= a_3-y_t; %10*1 column vector , error values
del_2=(Theta2' *del_3).*( a_2.*(1-a_2) );%26*1 column vector

del_2 = del_2(2:end);%removing the d0 term, now 25*1 vector
capDelta1= capDelta1+ del_2*a_1; %a_1=1*401 row vector, result=25*401 matrix
capDelta2= capDelta2 + del_3*a_2';%a_2' is 1*26 vector,result is 10*26 matrix

end


Theta1_grad= (1/m)*capDelta1;
Theta2_grad= (1/m)*capDelta2;

Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m)*Theta2(:,2:end);





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

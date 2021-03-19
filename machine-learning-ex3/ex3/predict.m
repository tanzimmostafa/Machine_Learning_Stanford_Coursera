function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(size(X,1),1) X];

z2 = Theta1*X';%25*5000 matrix where each column is another example and each row= a1(2),a2(2),a3(2)..i.e layer 2

ai_2 = sigmoid(z2);

ai_2 = [ones(1,size(ai_2,2));ai_2];%adding a row of 1s to ai_2 to include a0 terms in layer 2...now its a 26*5000 matrix

z3 = Theta2*ai_2; %10*5000 matrix 
  
h_theta=sigmoid(z3); %10*5000 matrix where each column is another example and each row is h_theta_x1,h_theta_x2..

[max_values,indices] =max(h_theta,[],1);%row vector of 5000elements

p = indices';%column vector containing elements from 1-10

% =========================================================================


end

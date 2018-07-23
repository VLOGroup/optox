clear all;


addpath('../../lib/matlab/');

x_rand = randn(64, 64, 32);
y_rand = randn(64, 64, 32, 3);

Ax = mex_nabla3_operator(x_rand);

ATy = mex_nabla3_operator_adjoint(y_rand);

lhs = sum(y_rand(:).*Ax(:))
rhs = sum(ATy(:).*x_rand(:))
abs(lhs - rhs)


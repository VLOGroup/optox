clear all;


addpath('../../lib/matlab/');

a = ones(3,3);
b = ones(3,3);

c = mex_add_operator(a, b)

[d, e] = mex_add_operator_adjoint(a)

exit


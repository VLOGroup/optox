#include <mex.h>


#include <iu/iucore.h>
#include <iu/iumatlab.h>

#include "operators/add_operator.h"

typedef double real_type;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // Checking number of arguments
  if (nrhs != 2)
    mexErrMsgIdAndTxt(
        "MATLAB:addoperator:invalidNumInputs",
        "2 inputs required (in_1, in_2)");
  if (nlhs > 1)
    mexErrMsgIdAndTxt("MATLAB:addoperator:maxlhs",
                      "Too many output arguments.");

  // parse input image
  iu::LinearHostMemory<real_type, 2> h_in1(*(prhs[0]), false);
  iu::LinearDeviceMemory<real_type, 2> d_in1(h_in1.size());
  iu::copy(&h_in1, &d_in1);
  iu::LinearHostMemory<real_type, 2> h_in2(*(prhs[1]), false);
  iu::LinearDeviceMemory<real_type, 2> d_in2(h_in2.size());
  iu::copy(&h_in2, &d_in2);

  // check whether the size matches
  if (d_in1.size()[0] != d_in2.size()[0] || d_in1.size()[1] != d_in2.size()[1])
    mexErrMsgIdAndTxt("MATLAB:addoperator:size",
                      "Inputs must have same size!");

  // output memory
  iu::LinearDeviceMemory<real_type, 2> d_out(d_in1.size());

  optox::AddOperator<real_type, 2> op;

	op.setParameter("w_1", 1.0);
	op.setParameter("w_2", 1.0);

	op.appendInput(d_in1);
	op.appendInput(d_in2);
	op.appendOutput(d_out);

	op.apply();

  iu::LinearHostMemory<real_type, 2> h_out(d_out.size());
  iu::copy(&d_out, &h_out);

  // push the output back to matlab
  iu::matlab::convertCToMatlab(h_out, &plhs[0]);
}

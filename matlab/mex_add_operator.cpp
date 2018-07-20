#include <mex.h>

#include <iu/iucore.h>
#include <iu/iumatlab.h>

#include "mex_utils.h"

#include "operators/add_operator.h"

typedef float T;

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
  std::unique_ptr<iu::LinearDeviceMemory<T, 2> > iu_in1 = getLinearDeviceFromMex<T, 2>(*(prhs[0]), false);
  std::unique_ptr<iu::LinearDeviceMemory<T, 2> > iu_in2 = getLinearDeviceFromMex<T, 2>(*(prhs[1]), false);

  // check whether the size matches
  if (iu_in1->size() != iu_in2->size())
    mexErrMsgIdAndTxt("MATLAB:addoperator:size",
                      "Inputs must have same size!");

  // output memory
  iu::LinearDeviceMemory<T, 2> iu_out(iu_in1->size());

  // setup the operator and apply it
  optox::AddOperator<T, 2> op;
	op.setParameter("w_1", 1.0);
	op.setParameter("w_2", 1.0);

	op.forward({&iu_out}, {iu_in1.get(), iu_in2.get()});

  // forward the result to matlab
  linearDeviceToMex(iu_out, &plhs[0], false);
}

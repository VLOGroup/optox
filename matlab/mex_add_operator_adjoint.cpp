#include <mex.h>

#include <iu/iucore.h>
#include <iu/iumatlab.h>

#include "mex_utils.h"

#include "operators/add_operator.h"

typedef float T;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // Checking number of arguments
  if (nrhs != 1)
    mexErrMsgIdAndTxt(
        "MATLAB:addoperatoradjoint:invalidNumInputs",
        "1 input required (in)");
  if (nlhs != 2)
    mexErrMsgIdAndTxt("MATLAB:addoperatoradjoint:maxlhs",
                      "Two output arguments required.");

  // parse input image
  std::unique_ptr<iu::LinearDeviceMemory<T, 2> > iu_in = getLinearDeviceFromMex<T, 2>(*(prhs[0]), false);

  // output memory
  iu::LinearDeviceMemory<T, 2> iu_out1(iu_in->size());
  iu::LinearDeviceMemory<T, 2> iu_out2(iu_in->size());

  // setup the operator and apply it
  optox::AddOperator<T, 2> op;
	op.setParameter("w_1", 1.0);
	op.setParameter("w_2", 1.0);

	op.adjoint({&iu_out1, &iu_out2}, {iu_in.get()});

  // forward the result to matlab
  linearDeviceToMex(iu_out1, &plhs[0], false);
  linearDeviceToMex(iu_out2, &plhs[1], false);
}

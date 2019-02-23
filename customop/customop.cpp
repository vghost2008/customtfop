#include <stdio.h>
#include <cfloat>
#include <iostream>
#include <boost/algorithm/clamp.hpp>
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using namespace std;
using namespace boost::algorithm;
typedef Eigen::ThreadPoolDevice CPUDevice;

/*
 * 输入tensor [X,Y,Z,...,M,N,..]tensor
 * 输入v[M,N,...] tensor
 * 输入index[num]，依次表示[X,Y,Z,...]维度的值
 * 将tensor中由index指定的值设置为
 * example:
 * tensor shape=[2,3,4,2,2]
 * v shape=[2,2]
 * index=[0,1,3]
 * 那么tensor[0,1,3]=v
 */
REGISTER_OP("MySetValue")
    .Attr("T: {int32,int64,float32,float64,bool}")
    .Input("tensor: T")
    .Input("v: T")
    .Input("index: int32")
	.Output("data:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class MySetValueOp: public OpKernel {
	public:
		explicit MySetValueOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_tensor        = context->input(0);
			const Tensor &_v             = context->input(1);
			const Tensor &_index         = context->input(2);
			auto          tensor         = _tensor.template flat<T>().data();
			auto          v              = _v.template flat<T>().data();
			auto          index          = _index.template flat<int>().data();
			auto          dim_nr         = _tensor.dims();
			auto          skip_dim_nr    = _index.dim_size(0);
			auto          offset         = 0;
			auto          block_size     = _v.NumElements();
			auto          cur_block_size = block_size;

			for(auto i=skip_dim_nr-1; i>=0; --i) {
				offset += index[i]*cur_block_size;
				cur_block_size *= _tensor.dim_size(i);
			}

			OP_REQUIRES(context, _index.dims()==1, errors::InvalidArgument("index must be 1-dimensional"));

			Tensor* output_data = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, _tensor.shape(), &output_data));

			output_data->CopyFrom(_tensor,_tensor.shape());

			auto      oq_tensor = output_data->template flat<T>().data();

			/*
			 * 如果原始数据中没有内容，使用0填充
			 */
			 copy(v,v+block_size,oq_tensor+offset);
		}
};
REGISTER_KERNEL_BUILDER(Name("MySetValue").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), MySetValueOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("MySetValue").Device(DEVICE_CPU).TypeConstraint<float>("T"), MySetValueOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("MySetValue").Device(DEVICE_CPU).TypeConstraint<double>("T"), MySetValueOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("MySetValue").Device(DEVICE_CPU).TypeConstraint<bool>("T"), MySetValueOp<CPUDevice, bool>);
REGISTER_KERNEL_BUILDER(Name("MySetValue").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), MySetValueOp<CPUDevice, tensorflow::int64>);

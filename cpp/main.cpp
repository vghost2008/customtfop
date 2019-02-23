#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include <dlfcn.h>

using namespace std;
using namespace tensorflow;

int main(int argc, char** argv) {
    auto h = dlopen("../customop/libcustomop.so",RTLD_NOW);
	cout<<h<<dlerror()<<endl;

	std::string PathGraph = "../output/frozen_graph.pb";
	tensorflow::Session* session;
	tensorflow::Status status;
	status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
	if (!status.ok()) {
		cout<<"AAA"<<endl;
		std::cout << status.ToString() << "\n";
		return 1;
	}
	tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({2,1}));
	std::vector<tensorflow::Tensor> output;
	auto input_tensor = input.template tensor<float,2>();
	input_tensor(0,0)= 1.0;
	input_tensor(1,0)= 2.0;
	tensorflow::GraphDef graph_def;
	status = ReadBinaryProto(tensorflow::Env::Default(),PathGraph, &graph_def);

	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	std::vector<std::pair<string,tensorflow::Tensor>> inputs = {
		{ "input:0", input},
	};
	status = session->Run(inputs, {"output"},{}, &output);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}
	auto Result = output[0].matrix<float>();
	float v0 = Result(0,0);
	float v1 = Result(1,0);

	std::cout << "Input: 0 | Output: "<< Result(0,0) << std::endl;
	std::cout << "Input: 1 | Output: "<< Result(1,0) << std::endl;
	char a;
	cin>>a;
} 

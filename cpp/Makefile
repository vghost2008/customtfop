TF_INC = -I/home/vghost/tensorflow/bazel-genfiles -I/home/vghost/tensorflow -I/home/vghost/tensorflow/bazel-tensorflow/external/eigen_archive -I/home/vghost/tensorflow/bazel-tensorflow/external/com_google_absl -I/home/vghost/tfinclude/a
TF_LIB = "/home/vghost/tensorflow/bazel-bin/tensorflow"
CXXFLAGS = -lm -Wl,--allow-multiple-definition -Wl,--whole-archive -Wl,--no-as-needed
CPPFLAGS = -lm -Wl,--allow-multiple-definition -Wl,--whole-archive -Wl,--no-as-needed
CFLAGS = -lm -Wl,--allow-multiple-definition -Wl,--whole-archive -Wl,--no-as-needed
all: clean build

build:run
run:main.cpp
	echo $(TF_LIB)
	g++ -std=c++14 -lm -Wl,--allow-multiple-definition -Wl,--whole-archive -Wno-unused-result -g -Wl,-rpath=${TF_LIB} -o $@ $^ $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -L${TF_LIB} -ltensorflow_cc -ltensorflow_framework -ldl 

clean:
	rm -f *.o *.so *.pyc *.npy run

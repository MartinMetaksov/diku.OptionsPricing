CXX        = nvcc -std=c++11 -O3 -Xcompiler -fopenmp


SOURCES_CPP =TestMain.cpp TestCuda.cu
EXECUTABLE  =../build/TestMain
COMPARE		=compare

default: compile

.cu.o: $(SOURCES_CPP)
	@$(CXX) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)

run: $(EXECUTABLE)
	@./$(EXECUTABLE)

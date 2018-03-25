CXX        = nvcc -std=c++11 --ptxas-options=-v


SOURCES_CPP =TestMain.cpp TestCudaOption.cu
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

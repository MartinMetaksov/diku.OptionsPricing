CXX        = nvcc -std=c++11 --maxrregcount=32 --ptxas-options=-v


SOURCES_CPP =CudaMain.cu
EXECUTABLE  =CudaMain
DATA		=../data
COMPARE		=compare

default: compile

.cu.o: $(SOURCES_CPP)
	@$(CXX) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)

compile-compare: $(COMPARE)
$(COMPARE):
	@futhark-opencl compare.fut

run: $(EXECUTABLE)
	@cat $(DATA)/$(FILE).in | ./$(EXECUTABLE)

test: $(EXECUTABLE)
	@cat $(DATA)/$(FILE).in | ./$(EXECUTABLE) -test

validate: $(EXECUTABLE) $(COMPARE)
	@echo "Is valid:"
	@cat $(DATA)/$(FILE).in | ./$(EXECUTABLE) > $(FILE).test.out
	@cat $(DATA)/$(FILE).out $(FILE).test.out | ./compare 

clean:
	@rm -f $(EXECUTABLE)



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
	@./$(EXECUTABLE) $(DATA)/$(FILE).in

test: $(EXECUTABLE)
	@./$(EXECUTABLE) $(DATA)/$(FILE).in -test

validate: $(EXECUTABLE) $(COMPARE)
	@echo "Is valid:"
	@./$(EXECUTABLE) $(DATA)/$(FILE).in > $(FILE).test.out
	@cat $(DATA)/$(FILE).out $(FILE).test.out | ./compare 

clean:
	@rm -f $(EXECUTABLE)



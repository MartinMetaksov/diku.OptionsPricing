CXX        = nvcc -std=c++11 --ptxas-options=-v


SOURCES_CPP =CudaOptionMain.cu
EXECUTABLE  =../build/CudaOption
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
	@futhark-opencl $(DATA)/compare.fut

run: $(EXECUTABLE)
	@./$(EXECUTABLE) $(DATA)/$(FILE).in

test: $(EXECUTABLE)
	@./$(EXECUTABLE) $(DATA)/$(FILE).in -test

mock: $(EXECUTABLE)
	@./$(EXECUTABLE) -test -mock $(COUNT)

validate: $(EXECUTABLE) $(COMPARE)
	@echo "Is valid:"
	@./$(EXECUTABLE) $(DATA)/$(FILE).in > $(FILE).test.out
	@cat $(DATA)/$(FILE).out $(FILE).test.out | ./$(DATA)/compare 

clean:
	@rm -f $(EXECUTABLE)



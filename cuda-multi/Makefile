CXX        = nvcc -std=c++11 #--maxrregcount=32 #--ptxas-options=-v
DEBUG_OPTIONS = -Xcompiler -rdynamic -lineinfo

SOURCES_CPP =CudaMultiMain.cu
EXECUTABLE  =../build/CudaMulti
DATA		=../data
COMPARE		=compare

default: compile

.cu.o: $(SOURCES_CPP)
	@$(CXX) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)

compile-debug: $(EXECUTABLE)
	@$(CXX) $(DEBUG_OPTIONS) -o $(EXECUTABLE) $(SOURCES_CPP)

run: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -s $(SORT) -v $(VERSION) -b $(BLOCK)

test: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -s $(SORT) -v $(VERSION) -b $(BLOCK) -t

repeat: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -s $(SORT) -v $(VERSION) -b $(BLOCK) -r 10

validate: $(EXECUTABLE) $(COMPARE)
	@echo "Is valid:"
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in > $(FILE).test.out
	@cat $(DATA)/$(FILE).out $(FILE).test.out | ./compare 

clean:
	@rm -f $(EXECUTABLE)



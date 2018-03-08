CXX        = g++ -g -std=c++11


SOURCES_CPP =SeqCutMain.cpp
EXECUTABLE  =SeqCutMain
DATA		=../data
COMPARE		=compare

default: compile

.o: $(SOURCES_CPP)
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
	@./$(EXECUTABLE) $(DATA)/$(FILE).in  > $(FILE).test.out
	@cat $(DATA)/$(FILE).out $(FILE).test.out | ./compare 

clean:
	@rm -f $(EXECUTABLE)



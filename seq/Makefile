CXX        = g++ -std=c++11


SOURCES_CPP =SeqMain.cpp
EXECUTABLE  =SeqMain
DATA		=../trinomial-option/data
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
	@cat $(DATA)/$(FILE).in | ./$(EXECUTABLE)

test: $(EXECUTABLE)
	@cat $(DATA)/$(FILE).in | ./$(EXECUTABLE) -test

validate: $(EXECUTABLE) $(COMPARE)
	@echo "Is valid:"
	@cat $(DATA)/$(FILE).in | ./$(EXECUTABLE) > $(FILE).test.out
	@cat $(DATA)/$(FILE).out $(FILE).test.out | ./compare 

clean:
	@rm -f $(EXECUTABLE)



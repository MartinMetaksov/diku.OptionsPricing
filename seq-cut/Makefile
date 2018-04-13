CXX        = g++ -g -std=c++11


SOURCES_CPP =SeqCutMain.cpp
EXECUTABLE  =../build/SeqCutMain
DATA		=../data
COMPARE		=compare

default: compile

.o: $(SOURCES_CPP)
	@$(CXX) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)

run: $(EXECUTABLE)
	@./$(EXECUTABLE) $(DATA)/$(FILE).in

test: $(EXECUTABLE)
	@./$(EXECUTABLE) $(DATA)/$(FILE).in -test

clean:
	@rm -f $(EXECUTABLE)



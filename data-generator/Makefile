CXX        = g++ -g -std=c++11


SOURCES_CPP =RandomDataGenerator.cpp
EXECUTABLE  =../build/RandomDataGenerator

default: compile

.o: $(SOURCES_CPP)
	@$(CXX) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)

run: $(EXECUTABLE)
	@./$(EXECUTABLE) -t $(TYPE) -h $(TOTAL_HEIGHT) -w $(TOTAL_WIDTH) -s $(SKEW_PERCENT)

clean:
	@rm -f $(EXECUTABLE)



CXX         	 = futhark-opencl

SOURCES_BASIC 	 = trinom-basic.fut
SOURCES_FLAT 	 = trinom-flat.fut
EXECUTABLE_BASIC = ../build/FutharkBasic
EXECUTABLE_FLAT  = ../build/FutharkFlat
DATA			 = ../data

compile-basic: $(EXECUTABLE_BASIC)
$(EXECUTABLE_BASIC): 
	@$(CXX) $(SOURCES_BASIC) -o $(EXECUTABLE_BASIC)

compile-flat: $(EXECUTABLE_FLAT)
$(EXECUTABLE_FLAT): 
	@$(CXX) $(SOURCES_FLAT) -o $(EXECUTABLE_FLAT)

run-basic: $(EXECUTABLE_BASIC)
	@cat $(DATA)/$(FILE).in $(DATA)/$(YIELD).in | ./$(EXECUTABLE_BASIC)

run-flat: $(EXECUTABLE_FLAT)
	@cat $(DATA)/$(FILE).in $(DATA)/$(YIELD).in | ./$(EXECUTABLE_FLAT)

clean:
	@rm -f $(EXECUTABLE_BASIC)
	@rm -f $(EXECUTABLE_BASIC).c
	@rm -f $(EXECUTABLE_FLAT)
	@rm -f $(EXECUTABLE_FLAT).c

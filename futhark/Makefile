CXX         	 = futhark-opencl

SOURCES_BASIC 	 = trinom-basic.fut
SOURCES_FLAT 	 = trinom-flat.fut
SOURCES_FLAT_OPT = trinom-flat-optimized.fut
EXECUTABLE_BASIC = ../build/FutharkBasic
EXECUTABLE_FLAT  = ../build/FutharkFlat
EXECUTABLE_FLAT_OPT  = ../build/FutharkFlatOptimized
DATA			 = ../data

compile-basic: $(EXECUTABLE_BASIC)
$(EXECUTABLE_BASIC): 
	@$(CXX) $(SOURCES_BASIC) -o $(EXECUTABLE_BASIC)

compile-flat: $(EXECUTABLE_FLAT)
$(EXECUTABLE_FLAT): 
	@$(CXX) $(SOURCES_FLAT) -o $(EXECUTABLE_FLAT)

compile-flat-opt: $(EXECUTABLE_FLAT_OPT)
$(EXECUTABLE_FLAT_OPT): 
	@$(CXX) $(SOURCES_FLAT_OPT) -o $(EXECUTABLE_FLAT_OPT)

run-basic: $(EXECUTABLE_BASIC)
	@cat $(DATA)/$(FILE).in $(DATA)/$(YIELD).in | ./$(EXECUTABLE_BASIC)

run-flat: $(EXECUTABLE_FLAT)
	@cat $(DATA)/$(FILE).in $(DATA)/$(YIELD).in | ./$(EXECUTABLE_FLAT)

run-flat-opt: $(EXECUTABLE_FLAT_OPT)
	@cat $(DATA)/$(FILE).in $(DATA)/$(YIELD).in | ./$(EXECUTABLE_FLAT_OPT)


clean:
	@rm -f $(EXECUTABLE_BASIC)
	@rm -f $(EXECUTABLE_BASIC).c
	@rm -f $(EXECUTABLE_FLAT)
	@rm -f $(EXECUTABLE_FLAT).c
	@rm -f $(EXECUTABLE_FLAT_OPT)
	@rm -f $(EXECUTABLE_FLAT_OPT).c

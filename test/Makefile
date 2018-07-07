CXX	= nvcc -std=c++11 -O3 -Xcompiler -fopenmp
CPP	= g++ -std=c++11 -O3	 

SOURCES_CPP = TestMain.cpp TestCuda.cu
EXECUTABLE  = ../build/Test

SOURCES_COMPARE		= Compare.cpp
EXECUTABLE_COMPARE  = ../build/Compare

ifeq ($(REAL),64)
  DEFINE_DOUBLE = -DUSE_DOUBLE
else
  DEFINE_DOUBLE =
endif

ifeq ($(REG),32)
  REG_COUNT = --maxrregcount=32
else
  REG_COUNT =
endif

default: run

.cu.o: $(SOURCES_CPP)
	@$(CXX) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) $(REG_COUNT) $(DEFINE_DOUBLE) -o $(EXECUTABLE) $(SOURCES_CPP)

compile-compare: $(EXECUTABLE_COMPARE)
$(EXECUTABLE_COMPARE):
	@$(CPP) $(DEFINE_DOUBLE) -o $(EXECUTABLE_COMPARE) $(SOURCES_COMPARE)

compare: $(EXECUTABLE_COMPARE)
	@cat $(DATA)/$(FILE1).out $(FILE2).out | ./$(EXECUTABLE_COMPARE)

run: $(EXECUTABLE)
	@./$(EXECUTABLE)

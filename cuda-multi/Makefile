CXX        	  = nvcc -std=c++11 -O3
DEBUG_OPTIONS = -G -DCUDA_ERROR_CHECK #--ptxas-options=-v

SOURCES_CPP = CudaMultiMain.cu
EXECUTABLE  = ../build/CudaMulti
DATA		= ../data
COMPARE		= compare

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

compile-debug: $(EXECUTABLE)
	@$(CXX) $(REG_COUNT) $(DEFINE_DOUBLE) $(DEBUG_OPTIONS) -o $(EXECUTABLE) $(SOURCES_CPP)

run: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -s $(SORT) -v $(VERSION) -b $(BLOCK) -d $(DEVICE)

test: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -s $(SORT) -v $(VERSION) -b $(BLOCK) -d $(DEVICE) -t

repeat: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -s $(SORT) -v $(VERSION) -b $(BLOCK) -d $(DEVICE) -t -r $(REP)

clean:
	@rm -f $(EXECUTABLE)



CXX        	  = nvcc -std=c++11 -O3 -arch=sm_35 #--ptxas-options=-v
DEBUG_OPTIONS = -G -DCUDA_ERROR_CHECK

SOURCES_CPP = CudaOptionMain.cu
EXECUTABLE  = ../build/CudaOption
DATA		= ../data

ifeq ($(REAL),64)
  DEFINE_DOUBLE = -DUSE_DOUBLE
else
  DEFINE_DOUBLE =
endif

ifeq ($(REG),32)
  REG_COUNT = --maxrregcount=32
else
	ifeq ($(REG),64)
	REG_COUNT = --maxrregcount=64
	else
	REG_COUNT =
	endif
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



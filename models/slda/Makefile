CC = g++
LDFLAGS = -lgsl -lm -lgslcblas -g


LSOURCE = main.cpp corpus.cpp slda.cpp utils.cpp opt.cpp dirichlet.c
LHEADER = corpus.h slda.h utils.h opt.h settings.h dirichlet.h

slda: $(LSOURCE) $(HEADER)
	  $(CC) $(LSOURCE) -o $@ $(LDFLAGS)

clean:
	-rm -f *.o slda

CC= g++
CFLAGS= -g -std=gnu++11
TARGETS= NeighborList NaiveNeighborList

all: $(TARGETS)

%: %.cpp
	$(CC) $(CFLAGS) $< -o $@
	rm -rf *.dSYM

clean:
	rm -f $(TARGETS)

CC = gcc
CFLAGS = -O3 -march=native -fopenmp -g #-Wno-unused-result
LDFLAGS = -lm -lfftw3

#CFLAG += -fopt-info-vec #-ffast-math
TARGET = ising_sim
SRC = ising_parallel_binary_double.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	@echo "Cleaning up build artifacts..."
	$(CC) $(CFLAGS) -c $< -o $@
        
clean:
	rm -f *.o $(TARGET)


run: $(TARGET)
	@echo "Running simulation..."
	./$(TARGET)
	@$(MAKE) clean

.PHONY: all clean

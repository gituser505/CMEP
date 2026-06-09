CC = gcc
CFLAGS = -O3 -march=native -fopenmp -g #-Wno-unused-result
LDFLAGS = -lm -lfftw3

#CFLAG += -fopt-info-vec #-ffast-math
TARGET = ising_sim
SRC = ising_parallel_binary_double.c
OBJ = $(SRC:.c=.o)

all: run

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@
        
clean:
	@echo "Cleaning up build artifacts..."
	rm -f *.o $(TARGET)

run: $(TARGET)
	@echo "Running simulation..."
	./$(TARGET)
	@$(MAKE) clean

.PHONY: all clean run
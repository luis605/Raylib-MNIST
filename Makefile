# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -I./raylib/src -g
LDFLAGS = -L./raylib/src -lraylib -lm -ldl -lpthread

# Target executable
TARGET = main

# Source files
SRCS = main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)
# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

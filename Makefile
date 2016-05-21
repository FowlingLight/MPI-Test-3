# compiler for the source files
CXX=mpic++

# flags to pass to the compiler when compiling files
# enables debugging and disables optimisation
CXXFLAGS=-g -O0 -std=c++11

# the list of object files that will compose your application
# these are basically the list of your .cpp files except with
# .cpp changed to .o
OBJECTS=file.o

# the target application that we are trying to build
TARGET=matrix_product

# the suffixes that this make file understands
.SUFFIXIES: .cpp .o

# the main target that will build the entire target
# note that the lines immediately after all: must start with
# exactly one tab stop
all: $(OBJECTS)
	$(CXX) -o $(TARGET) $(OBJECTS)

# rule to clean out all of our compiled files
clean:
	rm $(OBJECTS)
fclean:
	rm $(OBJECTS) $(TARGET)

# rule to convert a .cpp file into a .o file
.cpp.o:
	$(CXX) $(CXXFLAGS) -c $*.cpp

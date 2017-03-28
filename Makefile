CFLAGS=-g

COMMON_OBJECTS = cl_simple.o cl_util.o
LDFLAGS = -L/usr/local/lib -lOpenCL

TARGET=main
all: $(TARGET)

$(TARGET): $(TARGET).o $(COMMON_OBJECTS)
	gcc -o $@ $^ $(LDFLAGS)


CFLAGS := -fPIC
OBJECTS := tridiag.o tridiagv.o
libeigen.a : $(OBJECTS)
	ar cru $@ $^
	ranlib $@
clean:
	rm -f $(OBJECTS) libeigen.a
.PHONY : clean

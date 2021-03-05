# Specify extensions of files to delete when cleaning
CLEANEXTS   = o so

# Specify the source files, the target files,
SOURCES     = vdso.cc vdso_time.cc
OUTPUTFILE  = vdso.so

.PHONY: all
all: $(OUTPUTFILE)

$(OUTPUTFILE): $(subst .cc,.o,$(SOURCES))
	$(CC) $(CCFLAGS) -I. \
	-O2 -std=c++11 \
	-fPIC -fno-stack-protector -fuse-ld=gold -m64 -shared -nostdlib \
	-Wl,-soname=linux-vdso.so.1 \
	-Wl,--hash-style=sysv \
	-Wl,--no-undefined \
	-Wl,-T vdso_amd64.lds \
	-Wl,-Bsymbolic \
	-Wl,-z,max-page-size=4096 \
	-Wl,-z,common-page-size=4096 \
	-o vdso.so vdso.cc vdso_time.cc

.PHONY: clean
clean:
	for file in $(CLEANEXTS); do rm -f *.$$file; done

# Generate dependencies of .ccp files on .hpp files
include $(subst .cc,.d,$(SOURCES))

%.d: %.cc
	$(CC) -M $(CCFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
rm -f $@.$$$$


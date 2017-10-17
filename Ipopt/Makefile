# Copyright (C) 2005, 2010 International Business Machines and others.
# All Rights Reserved.
# This file is distributed under the Eclipse Public License.

# $Id: Makefile.in 2016 2011-06-15 08:53:32Z stefan $

##########################################################################
#    You can modify this example makefile to fit for your own program.   #
#    Usually, you only need to change the five CHANGEME entries below.   #
##########################################################################

# CHANGEME: This should be the name of your executable
EXE = RasterModel.exe

# CHANGEME: Here is the name of all object files corresponding to the source
#           code that you wrote in order to define the problem statement
OBJS = RasterModel_main.o RasterModelEuler_nlp.o RasterModelMidpoint_nlp.o Kernel.o RasterTools.o

# CHANGEME: Additional libraries
ADDLIBS =

# CHANGEME: Additional flags for compilation (e.g., include flags)
ADDINCFLAGS =

##########################################################################
#  Usually, you don't have to change anything below.  Note that if you   #
#  change certain compiler options, you might have to recompile Ipopt.   #
##########################################################################

# C++ Compiler command
CXX = x86_64-w64-mingw32-g++

# C++ Compiler options
CXXFLAGS = -O3 -pipe -fopenmp -DIPOPT_BUILD --std=c++11 -pg #-DNDEBUG

# additional C++ Compiler options for linking
CXXLINKFLAGS = 

# Include directories (we use the CYGPATH_W variables to allow compilation with Windows compilers)
#INCL = `PKG_CONFIG_PATH=/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib64/pkgconfig:/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib/pkgconfig:/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/share/pkgconfig:  --cflags ipopt` $(ADDINCFLAGS)
INCL = -I`$(CYGPATH_W) /c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/include/coin`  -I/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/include/coin/ThirdParty  -I/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/include/coin/ThirdParty    -I/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/include/coin/ThirdParty  -I/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/include/coin/ThirdParty          $(ADDINCFLAGS)

# Linker flags
#LIBS = `PKG_CONFIG_PATH=/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib64/pkgconfig:/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib/pkgconfig:/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/share/pkgconfig:  --libs ipopt`
#LIBS = -link -libpath:`$(CYGPATH_W) /c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib` libipopt.lib  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinmumps -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinmetis  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinblas -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32   -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinhsl -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinmetis  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinlapack -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinblas -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32   -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinlapack -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinblas -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32   -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinblas -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32  -lm 
LIBS = -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lipopt  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinmumps -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinmetis  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinblas -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32   -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinhsl -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinmetis  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinlapack -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinblas -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32   -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinlapack -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32  -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinblas -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32   -L/c/Users/Elliott/Documents/Ipopt-master/Ipopt-master/build/lib -lcoinblas -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0 -LC:/MinGW64/bin/../lib/gcc -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib/../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../../../x86_64-w64-mingw32/lib -LC:/MinGW64/bin/../lib/gcc/x86_64-w64-mingw32/5.1.0/../../.. -lgfortran -lmingw32 -lmoldname -lmingwex -lmsvcrt -lquadmath -lm -lpthread -ladvapi32 -lshell32 -luser32 -lkernel32  -lm 

# The following is necessary under cygwin, if native compilers are used
CYGPATH_W = echo

all: $(EXE)

.SUFFIXES: .cpp .c .o .obj

$(EXE): $(OBJS)
	bla=;\
	for file in $(OBJS); do bla="$$bla `$(CYGPATH_W) $$file`"; done; \
	$(CXX) $(CXXLINKFLAGS) $(CXXFLAGS) -o $@ $$bla $(ADDLIBS) $(LIBS)

clean:
	rm -rf $(EXE) $(OBJS) ipopt.out

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCL) -c -o $@ $<


.cpp.obj:
	$(CXX) $(CXXFLAGS) $(INCL) -c -o $@ `$(CYGPATH_W) '$<'`

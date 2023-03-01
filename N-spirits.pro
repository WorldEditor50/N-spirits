TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt


include($$PWD/basic/basic.pri)
include($$PWD/utils/utils.pri)
include($$PWD/net/net.pri)

SOURCES += \
        test.cpp

HEADERS += \
    bayes.h \
    hmm.h \
    kmeans.h \
    linearregression.h \
    svm.h

msvc {
    QMAKE_CFLAGS += /bigobj
    QMAKE_CXXFLAGS += /bigobj
    QMAKE_CFLAGS += /arch:AVX2
    QMAKE_CXXFLAGS += /arch:AVX2
    QMAKE_CXXFLAGS_DEBUG += -O2
    QMAKE_CFLAGS_DEBUG += -O2
}


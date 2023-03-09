TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt


include($$PWD/basic/basic.pri)
include($$PWD/Statistics/Statistics.pri)
include($$PWD/net/net.pri)

SOURCES += \
        test.cpp

include($$PWD/ml/ml.pri)

msvc {
    QMAKE_CFLAGS += /bigobj
    QMAKE_CXXFLAGS += /bigobj
    QMAKE_CFLAGS += /arch:AVX2
    QMAKE_CXXFLAGS += /arch:AVX2
    QMAKE_CXXFLAGS_DEBUG += -O2
    QMAKE_CFLAGS_DEBUG += -O2
}
JPEG_ROOT = D:/home/3rdparty/libjpeg
INCLUDEPATH += $$JPEG_ROOT/include
LIBS += -L$$JPEG_ROOT/lib -llibjpeg

TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt


include($$PWD/basic/basic.pri)
include($$PWD/utils/utils.pri)
include($$PWD/net/net.pri)

SOURCES += \
        main.cpp

HEADERS += \
    bayes.h \
    hmm.h \
    kmeans.h \
    linearregression.h \
    svm.h

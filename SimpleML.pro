TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        linearalgebra.cpp \
        main.cpp \
        mat.cpp

HEADERS += \
    basic_def.h \
    bayes.h \
    complexnumber.h \
    csv.h \
    dataset.h \
    fft.h \
    gmm.h \
    hmm.h \
    kernel.h \
    kmeans.h \
    linearalgebra.h \
    linearregression.h \
    mat.h \
    svm.h \
    utils.h \
    vec.h

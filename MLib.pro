TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        linearalgebra.cpp \
        main.cpp \
        mat.cpp

HEADERS += \
    bayes.h \
    csv.h \
    dataset.h \
    func.h \
    gmm.h \
    hmm.h \
    kernel.h \
    kmeans.h \
    linearalgebra.h \
    linearregression.h \
    mat.h \
    svm.h \
    vec.h

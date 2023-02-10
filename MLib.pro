TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
        mat.cpp

HEADERS += \
    bayes.h \
    csv.h \
    dataset.h \
    gmm.h \
    hmm.h \
    kernel.h \
    kmeans.h \
    linearregression.h \
    mat.h \
    svm.h \
    vec.h

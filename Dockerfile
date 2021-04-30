FROM gcc:9.3

RUN mkdir -p /TeraSort

WORKDIR /TeraSort

COPY ./TeraSort.cpp /TeraSort

RUN g++ -fopenmp -O3 -o prog TeraSort.cpp

RUN rm ./TeraSort.cpp

ENV PATH="/TeraSort:${PATH}"
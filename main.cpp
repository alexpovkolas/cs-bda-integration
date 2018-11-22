#include <iostream>
#include <mpi.h>
#include <map>
#include <string>
#include <vector>
#include <random>


using namespace std;

#define ROOT 0

void delegate(int rank, double start, double end, int blocks, int tag) {
    double buffer[3] = {start, end, (double) blocks};
    MPI_Send(buffer, 3, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
}

void free(int rank) {
    double buffer[1] = {0};
    MPI_Send(buffer, 0, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
}

void master_process(int world_size, int n, int blocks_count, double start, double end) {
    double step = (end - start) / n;
    double current = start;
    int current_tag = 0;
    double result = 0;
    int rank = 1;
    for (; rank < world_size; rank++) {
        delegate(rank, current, current + step, blocks_count, current_tag);
        current_tag++;
        current += step;
    }

    bool hasTask = true;
    while (current < end) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_DOUBLE, &count);


        double buffer[1];
        MPI_Recv(buffer, 1, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        result += buffer[0];

        rank = rank + 1 < world_size ? rank + 1 : 1;
        delegate(rank, current, current + step, blocks_count, current_tag);
        current_tag++;
        current += step;
    }

    for (int i=1; i < world_size; i++) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        double buffer[1];
        MPI_Recv(buffer, 1, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        result += buffer[0];

        //cout << "Current Result: " << result << endl;

        free(i);
    }

    cout << "Final Result: " << result << endl << endl;
}

double func(double arg) {
    return sin(arg);
}

double calculate(double start, double end, int blocks) {
    //cout << "calculate: " << start << " " << end << " " << blocks << endl;
    double step = (end - start) / blocks;
    double acc = 0;
    double arg = start;
    for (int i = 0; i < blocks; ++i) {
        acc += (func(arg) + func(arg + step)) * step / 2;
        arg += step;
    }
    return acc;
}

void slave_process(int rank) {
    bool hasTask = true;
    while (hasTask) {
        MPI_Status status;
        MPI_Probe(ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_DOUBLE, &count);

        if (count == 3) {
            double buffer[3];
            MPI_Recv(buffer, 3, MPI_DOUBLE, ROOT, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double result = calculate(buffer[0], buffer[1], buffer[2]);

            //cout << "Calculate Result: " << result << endl;

            // notify root
            MPI_Send(&result, 1, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD);
        } else {
            hasTask = false;
        }
    }
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int n = 1000;
    double start = 0;
    double end = 10;
    int blocks_count = 10;

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<string> result;
    if (rank == ROOT) {
        master_process(world_size, n, blocks_count, start, end);
    } else {
        slave_process(rank);
    }


    MPI_Finalize();
}


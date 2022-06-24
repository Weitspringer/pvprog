#include <iostream>
#include <cstdlib>
#include <cmath>
#include <pthread.h>

using namespace std;

#define NUM_THREADS 5

struct threadArgs {
    int x;
    int y;
    vector<vector<double>> heatmap;
}

double *CalculateTemperatureFromNeighborsAndSelf(int x, int y, vector<vector<double>> &heatmap){
    int numRows = heatmap.size();
    int numCols = heatmap[0].size();

    // Noch nicht beachtet: Hotspots über Zeit

    double sum = 0;
    sum += heatmap[x][y]; // self
    sum += (x>=0 && y>=0) ? heatmap[x-1][y-1] : 0 // nordwest
    sum += (x>=0) ? heatmap[x-1][y] : 0; // west
    sum += (x>=0 && y<numRows) ? heatmap[x-1][y+1] : 0 // suedwest
    sum += (y<numRows) ? heatmap[x][y+1] : 0 // sued
    sum += (x<numCols && y<numRows) ? heatmap[x+1][y+1] : 0 // suedost
    sum += (x<numCols) ? heatmap[x+1][y] : 0 // ost
    sum += (x<numCols && y>=0) ? heatmap[x+1][y-1] : 0 // nordost
    sum += (y>=0) ? heatmap[x][y-1] : 0 // nord
    
    return (double) sum/9;
}

void *simulateStep(vector<vector<double>> &heatmap) {
    vector<vector<double>> nextRoundHeatMap = heatmap;

    int batches = ceil(heatmap.size() * heatmap[0].size() / NUM_THREADS);

    int x = 0;
    int y = 0;

    for (int i = 1; i <= batches; i++) {
        for (t = 0; t < NUM_THREADS; t++) {
            printf("In main: creating thread %ld\n", t);
            rc = pthread_create(&threads[t], NULL, CalculateTemperatureFromNeighborsAndSelf, (void *)t);
            if (rc != 0) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }

        for (t = 0; t < NUM_THREADS; t++) {
            rc = pthread_join(threads[t], NULL);
            if (rc != 0) {
                printf("ERROR; return code from pthread_join() is %d\n", rc);
                exit(-1);
            }
        }
    }
}

int main (int argc, char** argv) {

    int fieldWidth = argv[0];
    int fieldHeight = argv[1];
    int numberOfRounds = argv[2];
    string hotspotFileName = argv[3];

    vector<vector<double>> heatmap(fieldWidth, vector<double>(fieldHeight, 0));

    pthread_t threads[NUM_THREADS];
    int rc;
    int i;
    
    for( i = 0; i < NUM_THREADS; i++ ) {
        cout << "main() : creating thread, " << i << endl;
        rc = pthread_create(&threads[i], NULL, PrintHello, (void *)i);
        
        if (rc) {
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
        }
    }

    for(int roundNum = 1; roundNum <= numberOfRounds; roundNum++) {
        
        simulateStep(heatmap);
    }

    pthread_exit(NULL);
}
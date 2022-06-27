#include "karpFlattMetric.h"

void readMeasurementData(string filename, Measurement& measurements)
{
    if (filename != "")
    {
        bool flag_header_read = false;

        ifstream raw_data_file(filename);
        if (raw_data_file.is_open())
        { // always check whether the file is open
            while (raw_data_file)
            {
                string line;
                getline(raw_data_file, line);
                if (flag_header_read)
                {
                    vector<string> tokens;
                    string token;
                    istringstream tokenStream(line);
                    if (!line.empty())
                    {
                        while (getline(tokenStream, token, ','))
                        {
                            tokens.push_back(token);
                        }
                        int strategy = stoi(tokens[0]);
                        int numThreads = stoi(tokens[1]);
                        string numPoints = token[2];

                        measurements.addKey(strategy, numThreads, numPoints);
                        measurements.addpossibleThreadNum(numThreads) 
                        measurements.addMeasurement(strategy, numThreads, numPoints, stod(tokens[3]));
                    }
                }
                else
                {
                    flag_header_read = true;
                }
            }
        }
        else
        {
            cout << "Couldn't open file\n";
        }
    }
}

double calculateKarpFlattMetric(int strategy, int64_t numPoints, Measurement &measurement) {
    double serialFraction = 1;

    for(auto const& numThread : measurement.getPossibleThreads())

    int nthreads = 0;
    if (nthreads > 1) {
        double serialFraction = ((1 / speedup) - (1 / nthreads)) / (1 - (1 / nthreads));
    } else {
        cout << "Error calculating Karp-Flatt metric: nthreads must be greater than 1" << endl;
    }
    return serialFraction;
}


int main(int argc, char* argv[])
{
	string filename = "measurements.csv";
    Measurement measurements;
    readMeasurementData(filename, measurements);
    //measurements.print();
    //calculateKarpFlattMetric()
}

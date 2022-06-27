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
                    vector<int> tokens;
                    string token;
                    istringstream tokenStream(line);
                    if (!line.empty())
                    {
                        while (getline(tokenStream, token, ','))
                        {
                            tokens.push_back(stoi(token));
                        }
                        measurements.addValue(pair(tokens))
                        lifecycles.addValue(tokens[0], tokens[1], tokens[2], tokens[3]);
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

double calculateKarpFlattMetric(double speedup, short nthreads) {
    double serialFraction = 1;
    if (nthreads > 1) {
        double serialFraction = ((1 / speedup) - (1 / nthreads)) / (1 - (1 / nthreads));
    } else {
        perror("Error calculating Karp-Flatt metric: nthreads is not greater than 1");
    }
    return serialFraction;
}
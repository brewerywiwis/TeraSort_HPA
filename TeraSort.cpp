#include <omp.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

#define TASK_SIZE 10000
#define MAX_LENGTH 5120    //5KB
#define CHUNK_SIZE 524288  //512KB
const int batch_size = 20000000;
const string tmp_prefix = "tmp_out_";
const string tmp_suffix = ".txt";

pair<string, double>* memory = new pair<string, double>[batch_size];
char* line = new char[MAX_LENGTH + 64];
char file_buffer[CHUNK_SIZE + 64];

struct HeapNode {
    pair<string, double> value;
    int chunk_num;
};

struct arguments {
    pair<string, double>* values;
    size_t size;
    size_t chunk;
};

struct MinHeapCompare {
    bool operator()(HeapNode& n1, HeapNode& n2) {
        return n1.value.second >= n2.value.second;  // Ascending order
    }
};

bool pairCompare(pair<string, double>& p1, pair<string, double>& p2) {
    return p1.second < p2.second;  // Ascending order
}

void merge(pair<string, double> arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    pair<string, double>* L = new pair<string, double>[n1];
    pair<string, double>* R = new pair<string, double>[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i].second <= R[j].second) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    delete[] L;
    delete[] R;
}

void mergeSort(pair<string, double> arr[], int l, int r) {
    if (l >= r) {
        return;
    }
    int m = l + (r - l) / 2;
    int n = 1 + r - l;
    if (n >= 32) {
#pragma omp taskgroup
        {
#pragma omp task shared(arr) untied if (n > TASK_SIZE)
            mergeSort(arr, l, m);

#pragma omp task shared(arr) untied if (n > TASK_SIZE)
            {
                mergeSort(arr, m + 1, r);
            }
#pragma omp taskyield
        }
        merge(arr, l, m, r);
    } else {
        sort(arr + l, arr + r + 1, pairCompare);
    }
}

void* write_vals(void* args) {
    // tmp_out_1.txt,  tmp_out_2.txt ...
    arguments* params = (arguments*)args;
    string output_file = (tmp_prefix + to_string(params->chunk) + tmp_suffix);
    FILE* pFile = fopen(output_file.c_str(), "wb");
    int buffer_count = 0;
    int i = 0;
    while (i < params->size) {
        buffer_count += sprintf(&file_buffer[buffer_count], "%s: %f\n", params->values[i].first.c_str(), params->values[i].second);
        i++;

        // if the chunk is big enough, write it.
        if (buffer_count >= CHUNK_SIZE) {
            fwrite(file_buffer, buffer_count, 1, pFile);
            buffer_count = 0;
        }
    }
    if (buffer_count > 0) {
        fwrite(file_buffer, buffer_count, 1, pFile);
    }
    cout << output_file << " is created.\n";
    fclose(pFile);
    free(args);
    return NULL;
}

string mergeFiles(int chunks, const string& merge_file) {
    FILE* output = fopen(merge_file.c_str(), "wb");
    priority_queue<HeapNode, vector<HeapNode>, MinHeapCompare> minHeap;

    FILE** files = new FILE*[chunks];

    for (int i = 1; i <= chunks; i++) {
        // generate a unique name for temp file (temp_out_1.txt , temp_out_2.txt ..)
        string sorted_file = (tmp_prefix + to_string(i) + tmp_suffix);

        files[i - 1] = fopen(sorted_file.c_str(), "r");

        if (fgets(line, MAX_LENGTH, files[i - 1])) {
            // std::cout << line << endl;
            string name;
            double num;

            istringstream ss(line);
            ss >> name >> num;
            name = name.substr(0, name.size() - 1);
            // std::cout << name << " " << to_string(num) << endl;
            HeapNode top{pair<string, double>{name, num}, (i - 1)};

            minHeap.push(top);
        };
    }

    while (minHeap.size() > 0) {
        HeapNode min_node = minHeap.top();

        minHeap.pop();

        char line_buffer[100];
        int buffer_size = sprintf(line_buffer, "%s: %f\n", min_node.value.first.c_str(), min_node.value.second);
        fwrite(line_buffer, buffer_size, 1, output);

        if (fgets(line, MAX_LENGTH, files[min_node.chunk_num])) {
            string name;
            double num;

            istringstream ss(line);
            ss >> name >> num;
            name = name.substr(0, name.size() - 1);

            HeapNode heap_node{pair<string, double>{name, num}, min_node.chunk_num};

            minHeap.push(heap_node);
        }
    }

    for (int i = 1; i <= chunks; i++) {
        fclose(files[i - 1]);
    }

    fclose(output);
    delete[] files;
    cout << "<<MERGE COMPLETED>>\n";
    return merge_file;
}
int main(int argc, char** argv) {
    string outputFile;
    string inputFile;
    if (argc == 3) {
        outputFile = argv[2];
        inputFile = argv[1];
        cout << "<<Command is correct and the processes will be started>>\n";
    } else {
        cout << "Please insert arguments in format: " << argv[0] << " <input> <output>\n";
        return 1;
    }
    FILE* in = fopen(inputFile.c_str(), "r");
    bool batch_success = false;
    int current_batch = 1;
    int current_size = 0;

    omp_set_num_threads(omp_get_max_threads() * 6);

    while (fgets(line, MAX_LENGTH, in)) {
        batch_success = false;

        string name;
        double num;

        istringstream ss(line);
        ss >> name >> num;
        name = name.substr(0, name.size() - 1);

        memory[current_size++] = pair<string, double>{name, num};

        if (current_size >= batch_size) {
#pragma omp parallel
            {
#pragma omp single
                mergeSort(memory, 0, current_size - 1);
            }

            struct arguments* p = (struct arguments*)malloc(sizeof(struct arguments));
            p->values = memory;
            p->size = current_size;
            p->chunk = current_batch;

            write_vals(p);

            current_batch++;
            current_size = 0;
            batch_success = true;
        }
    }

    if (!batch_success) {
#pragma omp parallel
        {
#pragma omp single
            mergeSort(memory, 0, current_size - 1);
        }
        struct arguments* p = (struct arguments*)malloc(sizeof(struct arguments));
        p->values = memory;
        p->size = current_size;
        p->chunk = current_batch;

        write_vals(p);

    } else {
        current_batch--;
    }
    fclose(in);
    cout << "<<CHECKPOINT before merging tmp files>>\n";
    delete[] memory;

    if (current_batch == 0) {
        cout << "no data found\n";
    } else {
        mergeFiles(current_batch, outputFile);
        cout << "Sorted output is in file: " << outputFile << "\n";
    }

    return 0;
}
#include "../../hnswlib/hnswlib.h"
#include <set>


int main() {
    int dim = 200000;//16;               // Dimension of the elements
    int max_elements = 1000;//10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 20;  // Controls index search speed/build speed tradeoff

    int none_zero_num_bound = 100;
    int max_value = 10;

    // Initing index
    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    alg_hnsw->setEf(100);

    // generate sparse data
    srand((unsigned)time(NULL));
    char **data = new char*[max_elements];
    for (int i = 0; i < max_elements; i++)
    {
        //generate none_zero_num
        hnswlib::vectorsizeint none_zero_num = rand() % none_zero_num_bound + 1;
        data[i] = new char[none_zero_num * (sizeof(hnswlib::vectordata_t) + \
            sizeof(hnswlib::vectorsizeint)) + sizeof(hnswlib::vectorsizeint)];
        hnswlib::vectorsizeint len = none_zero_num;
        memcpy(data[i], &len, sizeof(hnswlib::vectorsizeint));
        //generate data
        for (int j = 0; j < none_zero_num; j++)
        {
            hnswlib::vectordata_t value = rand() % max_value + 1;
            memcpy(data[i] + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
                &value, sizeof(hnswlib::vectordata_t));
        }
        //generate index
        std::set<hnswlib::vectorsizeint> index_set;
        while (index_set.size() < none_zero_num)
        {
            index_set.insert(rand() % dim);
        }
        std::set<hnswlib::vectorsizeint>::iterator iter = index_set.begin();
        for (int k = 0; k < none_zero_num; k++)
        {
            hnswlib::vectorsizeint index = *iter;
            memcpy(data[i] + sizeof(hnswlib::vectorsizeint) + none_zero_num * sizeof(hnswlib::vectordata_t) \
                + k * sizeof(hnswlib::vectorsizeint), &index, sizeof(hnswlib::vectorsizeint));
            iter++;
        }
    }

    // Normalize data
    for (int i = 0; i < max_elements; i++)
    {
        float norm = 0;
        hnswlib::vectorsizeint len;
        memcpy(&len, data[i], sizeof(hnswlib::vectorsizeint));
        for (int j = 0; j < len; j++)
        {
            hnswlib::vectordata_t value;
            memcpy(&value, data[i] + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
                sizeof(hnswlib::vectordata_t));
            norm += value * value;
        }
        norm = sqrt(norm);
        for (int j = 0; j < len; j++)
        {
            hnswlib::vectordata_t value;
            memcpy(&value, data[i] + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
                sizeof(hnswlib::vectordata_t));
            value /= norm;
            memcpy(data[i] + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
                &value, sizeof(hnswlib::vectordata_t));
        }
    }

    clock_t start1, finish1;
    clock_t start2, finish2;

    // Add data to index
    start1 = clock();
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data[i], i);
    }
    finish1 = clock();
    std::cout << "addPoint finish\n";
    
    // Query the elements for themselves and measure recall
    start2 = clock();
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::vector<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnCloserFirst(data[i], 3);
        hnswlib::labeltype label = result[0].second;
        float dist = result[0].first;
        if (label == i) correct++;
        if (label != i){
            std::cout << "i: " << i << "\n";
            std::cout << "label: " << label << "\n";
            std::cout << "dist: " << dist << "\n";

        }
    }
    finish2 = clock();

    float recall = correct / max_elements;
    float time1 = (float)(finish1 - start1) / CLOCKS_PER_SEC;
    float time2 = (float)(finish2 - start2) / CLOCKS_PER_SEC;
    std::cout << "Recall: " << recall << "\n";
    std::cout << "Time of addPoint: " << time1 << "\n";
    std::cout << "Time of searchKnn: " << time2 << "\n";

    // // Serialize index
    // std::string hnsw_path = "hnsw.bin";
    // alg_hnsw->saveIndex(hnsw_path);
    // delete alg_hnsw;

    // // Deserialize index and check recall
    // alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    // correct = 0;
    // for (int i = 0; i < max_elements; i++) {
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second;
    //     if (label == i) correct++;
    // }
    // recall = (float)correct / max_elements;
    // std::cout << "Recall of deserialized index: " << recall << "\n";

    for (int i = 0; i < max_elements; i++)
    {
        delete[] data[i];
    }
    delete[] data;
    delete alg_hnsw;
    
    return 0;
}

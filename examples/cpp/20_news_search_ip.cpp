#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/load_data.h"
#include <set>


int main() {
    int dim = 173762;//16;               // Dimension of the elements
    int max_elements = 18646;//10000;   // Maximum number of elements, should be known beforehand
    int M = 64;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 100;  // Controls index search speed/build speed tradeoff

    // int none_zero_num_bound = 100;
    // int max_value = 10;
    int query_num = 200;
    int gt_num = 100;
    int top_k = 10;

    // Initing index
    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    alg_hnsw->setEf(10);

    // load base data
    char **data = new char*[max_elements];
    char basedata_file[100] = "../../test_data/20news/20_newsgroups_basedata_sparse.bin";
    LoadBinToSparseVector<hnswlib::vectordata_t, hnswlib::vectorsizeint>(basedata_file, \
        data, max_elements, dim);
    
    // load query
    char **query = new char*[query_num];
    char query_file[100] = "../../test_data/20news/20_newsgroups_querydata_sparse.bin";
    LoadBinToSparseVector<hnswlib::vectordata_t, hnswlib::vectorsizeint>(query_file, \
        query, query_num, dim);
    
    // load ground truth
    hnswlib::vectorsizeint *gt = new hnswlib::vectorsizeint[query_num * gt_num];
    char gt_file[100] = "../../test_data/20news/20_newsgroups_gt.bin";
    LoadBinToArray<hnswlib::vectorsizeint>(gt_file, gt, query_num, gt_num);
    hnswlib::vectordata_t *gt_dis = new hnswlib::vectordata_t[query_num * gt_num];
    char gt_dis_file[100] = "../../test_data/20news/20_newsgroups_gt_dis.bin";
    LoadBinToArray<hnswlib::vectordata_t>(gt_dis_file, gt_dis, query_num, gt_num);

    // // generate sparse data
    // srand((unsigned)time(NULL));
    // char **data = new char*[max_elements];
    // for (int i = 0; i < max_elements; i++)
    // {
    //     //generate none_zero_num
    //     hnswlib::vectorsizeint none_zero_num = 11;//rand() % none_zero_num_bound + 1;
    //     data[i] = new char[none_zero_num * (sizeof(hnswlib::vectordata_t) + \
    //         sizeof(hnswlib::vectorsizeint)) + sizeof(hnswlib::vectorsizeint)];
    //     hnswlib::vectorsizeint len = none_zero_num;
    //     memcpy(data[i], &len, sizeof(hnswlib::vectorsizeint));
    //     //generate data
    //     for (int j = 0; j < none_zero_num; j++)
    //     {
    //         hnswlib::vectordata_t value = rand() % max_value + 1;
    //         memcpy(data[i] + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
    //             &value, sizeof(hnswlib::vectordata_t));
    //     }
    //     //generate index
    //     std::set<hnswlib::vectorsizeint> index_set;
    //     while (index_set.size() < none_zero_num)
    //     {
    //         index_set.insert(rand() % dim);
    //     }
    //     std::set<hnswlib::vectorsizeint>::iterator iter = index_set.begin();
    //     for (int k = 0; k < none_zero_num; k++)
    //     {
    //         hnswlib::vectorsizeint index = *iter;
    //         memcpy(data[i] + sizeof(hnswlib::vectorsizeint) + none_zero_num * sizeof(hnswlib::vectordata_t) \
    //             + k * sizeof(hnswlib::vectorsizeint), &index, sizeof(hnswlib::vectorsizeint));
    //         iter++;
    //     }
    // }

    // // Normalize data
    // for (int i = 0; i < max_elements; i++)
    // {
    //     float norm = 0;
    //     hnswlib::vectorsizeint len;
    //     memcpy(&len, data[i], sizeof(hnswlib::vectorsizeint));
    //     for (int j = 0; j < len; j++)
    //     {
    //         hnswlib::vectordata_t value;
    //         memcpy(&value, data[i] + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
    //             sizeof(hnswlib::vectordata_t));
    //         norm += value * value;
    //     }
    //     norm = sqrt(norm);
    //     for (int j = 0; j < len; j++)
    //     {
    //         hnswlib::vectordata_t value;
    //         memcpy(&value, data[i] + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
    //             sizeof(hnswlib::vectordata_t));
    //         value /= norm;
    //         memcpy(data[i] + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
    //             &value, sizeof(hnswlib::vectordata_t));
    //     }
    // }

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
    for (int test_i = 0; test_i < 10; test_i++) {
        alg_hnsw->setEf(10 + test_i * 5);
        printf("ef = %d\n", 10 + test_i * 5);
        start2 = clock();
        float correct = 0;
        for (int i = 0; i < query_num; i++) {
            std::vector<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnCloserFirst(query[i], top_k);
            for (int j = 0; j < top_k; j++) {
                for (int k = 0; k < top_k; k++) {
                    if (result[j].second == gt[i * gt_num + k]) {
                        correct += 1;
                        break;
                    }
                }
            }
        }
        finish2 = clock();

        float recall = correct / (query_num * top_k);
        float time1 = (float)(finish1 - start1) / CLOCKS_PER_SEC;
        float time2 = (float)(finish2 - start2) / CLOCKS_PER_SEC;
        std::cout << "Recall: " << recall << "\n";
        std::cout << "Time of addPoint: " << time1 << "\n";
        std::cout << "Time of searchKnn: " << time2 << "\n";
    }

    // Serialize index
    char hnsw_path[100];
    sprintf(hnsw_path, "20news_ef_%d_M_%d.bin", ef_construction, M);
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    float correct = 0;
    for (int i = 0; i < query_num; i++) {
        std::vector<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnCloserFirst(query[i], top_k);
        for (int j = 0; j < top_k; j++) {
            for (int k = 0; k < top_k; k++) {
                if (result[j].second == gt[i * gt_num + k]) {
                    correct += 1;
                    break;
                }
            }
        }
    }
    float recall = correct / (query_num * top_k);
    std::cout << "Recall of deserialized index: " << recall << "\n";

    for (int i = 0; i < max_elements; i++)
    {
        delete[] data[i];
    }
    delete[] data;
    delete alg_hnsw;
    
    return 0;
}

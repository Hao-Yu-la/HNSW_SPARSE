#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

typedef unsigned int vectorsizeint_;
typedef float vectordata_t_;


float InnerProduct(const void *pVect1, const void *pVect2) {
    vectorsizeint_ len1 = *((vectorsizeint_ *) pVect1);
    vectorsizeint_ len2 = *((vectorsizeint_ *) pVect2);

    vectordata_t_ *data1 = (vectordata_t_ *) ((vectorsizeint_ *) pVect1 + 1);
    vectordata_t_ *data2 = (vectordata_t_ *) ((vectorsizeint_ *) pVect2 + 1);
    vectorsizeint_ *index1 = (vectorsizeint_ *) (data1 + len1);
    vectorsizeint_ *index2 = (vectorsizeint_ *) (data2 + len2);

    float res = 0;
    while ((char *)index1 < (char *) pVect1 + len1 * (sizeof(vectordata_t_) + sizeof(vectorsizeint_)) + sizeof(vectorsizeint_)
        && (char *)index2 < (char *) pVect2 + len2 * (sizeof(vectordata_t_) + sizeof(vectorsizeint_)) + sizeof(vectorsizeint_)) {
        if (*index1 == *index2) {
            res += *data1 * *data2;
            data1++;
            data2++;
            index1++;
            index2++;
        } else if (*index1 < *index2) {
            data1++;
            index1++;
        } else {
            data2++;
            index2++;
        }
    }
    return res;
}

int main(){
    vectordata_t_ *data1 = new vectordata_t_[8];
    vectorsizeint_ *index1 = new vectorsizeint_[8];
    vectorsizeint_ len1 = 8;
    vectordata_t_ *data2 = new vectordata_t_[6];
    vectorsizeint_ *index2 = new vectorsizeint_[6];
    vectorsizeint_ len2 = 6;

    for (size_t i = 0; i < 8; i++)
    {
        data1[i] = i;
        index1[i] = i;
    }
    for (size_t i = 0; i < 6; i++)
    {
        data2[i] = 2*i;
        index2[i] = 2*i;
    }

    void *pVect1 = malloc(8 * (sizeof(vectordata_t_) + sizeof(vectorsizeint_)) + sizeof(vectorsizeint_));
    void *pVect2 = malloc(6 * (sizeof(vectordata_t_) + sizeof(vectorsizeint_)) + sizeof(vectorsizeint_));
    memcpy(pVect1, &len1, sizeof(vectorsizeint_));
    memcpy((char *)pVect1 + sizeof(vectorsizeint_), data1, 8 * sizeof(vectordata_t_));
    memcpy((char *)pVect1 + sizeof(vectorsizeint_) + 8 * sizeof(vectordata_t_), index1, 8 * sizeof(vectorsizeint_));
    memcpy(pVect2, &len2, sizeof(vectorsizeint_));
    memcpy((char *)pVect2 + sizeof(vectorsizeint_), data2, 6 * sizeof(vectordata_t_));
    memcpy((char *)pVect2 + sizeof(vectorsizeint_) + 6 * sizeof(vectordata_t_), index2, 6 * sizeof(vectorsizeint_));

    std::cout << InnerProduct(pVect1, pVect2) << std::endl;
    free (pVect1);
    free (pVect2);
    delete[] data1;
    delete[] index1;
    delete[] data2;
    delete[] index2;

    return 0; 
}
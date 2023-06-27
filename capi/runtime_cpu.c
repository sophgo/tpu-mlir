#include<stdlib.h>
#include<stdio.h>
#include<assert.h>

typedef long int intptr_t;
// Define Memref Descriptor
typedef struct MemrefDescriptor {
    float* allocated;
    float* aligned;
    intptr_t offset;
    intptr_t sizes[4];
    intptr_t strides[4];
}Memref;

// C interface of model
// input: arg1  output: arg0
void _mlir_ciface_model(Memref* arg0, Memref* arg1);

int main(int argc, char** argv){
    Memref* arg0 = (Memref*)malloc(sizeof(Memref));
    Memref* arg1 = (Memref*)malloc(sizeof(Memref));

    FILE* data_in = fopen("data_for_capi.txt","r");
    
    fscanf(data_in, "%ld", &arg1->sizes[0]);
    fscanf(data_in, "%ld", &arg1->sizes[1]);
    fscanf(data_in, "%ld", &arg1->sizes[2]);
    fscanf(data_in, "%ld", &arg1->sizes[3]);

    intptr_t in_size;
    fscanf(data_in, "%ld", &in_size);
    arg1->aligned = (float*)malloc(sizeof(float) * in_size);
    for(int i = 0; i < in_size; i++){
        float a;
        fscanf(data_in, "%f", &a);
        arg1->aligned[i] = a;
    }
    fclose(data_in);

    _mlir_ciface_model(arg0, arg1);

    char name[100] = "inference_result.txt";
    FILE* data_out = fopen(name, "w");
    assert(argc == 2);    //> a.out out_size
    intptr_t out_size = atol(argv[1]);
    for(int i = 0; i < out_size; i++) {
        fprintf(data_out, "%f\n", arg0->aligned[i]);
    }
    fclose(data_out);
    return 0;
}
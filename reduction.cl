__kernel void reduction(
                   __global int* out,
                   int size,
                   __global int* input)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    int sum = 0;

    for (int i = 0; i < size * size; i++){
             sum -= input[i];
    }

    *out = sum;
}

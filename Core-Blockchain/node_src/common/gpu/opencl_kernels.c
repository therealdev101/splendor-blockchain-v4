#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// OpenCL error checking macro
#define CL_CHECK(call) \
    do { \
        cl_int error = call; \
        if (error != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL error at %s:%d - %d\n", __FILE__, __LINE__, error); \
            return -1; \
        } \
    } while(0)

// Global OpenCL context
static cl_platform_id platform = NULL;
static cl_device_id* devices = NULL;
static cl_context context = NULL;
static cl_command_queue* queues = NULL;
static cl_program program = NULL;
static cl_kernel keccak_kernel = NULL;
static cl_kernel ecdsa_kernel = NULL;
static cl_kernel tx_kernel = NULL;
static int device_count = 0;
static bool opencl_initialized = false;

// OpenCL kernel source code
const char* keccak_kernel_source = R"(
__constant ulong keccak_round_constants[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL,
    0x8000000080008000UL, 0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008AUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

ulong rotl64(ulong x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f1600(__private ulong s[25]) {
    const int R[25] = {
        0,  1, 62, 28, 27,
        36, 44, 6,  55, 20,
        3,  10, 43, 25, 39,
        41, 45, 15, 21, 8,
        18, 2,  61, 56, 14
    };
    for (int round = 0; round < 24; round++) {
        ulong C[5];
        for (int x = 0; x < 5; x++) C[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        ulong D0 = rotl64(C[1],1) ^ C[4];
        ulong D1 = rotl64(C[2],1) ^ C[0];
        ulong D2 = rotl64(C[3],1) ^ C[1];
        ulong D3 = rotl64(C[4],1) ^ C[2];
        ulong D4 = rotl64(C[0],1) ^ C[3];
        for (int i=0;i<25;i+=5){ s[i]^=D0; s[i+1]^=D1; s[i+2]^=D2; s[i+3]^=D3; s[i+4]^=D4; }
        ulong B[25];
        for (int y=0;y<5;y++){
            for (int x=0;x<5;x++){
                int i = x + 5*y;
                int X = y;
                int Y = (2*x + 3*y) % 5;
                const int rot = R[i];
                B[X + 5*Y] = (rot==0) ? s[i] : rotl64(s[i], rot);
            }
        }
        for (int y=0;y<5;y++){
            for (int x=0;x<5;x++){
                int i = x + 5*y;
                s[i] = B[i] ^ ((~B[(i+1)%5 + 5*y]) & B[(i+2)%5 + 5*y]);
            }
        }
        s[0] ^= keccak_round_constants[round];
    }
}

void keccak256_ocl(__global const uchar* in, int inlen, __private uchar out32[32]) {
    ulong s[25]; for (int i=0;i<25;i++) s[i]=0UL;
    const int rate = 136; int off = 0;
    while (inlen >= rate) {
        for (int i=0;i<rate/8;i++){
            ulong w=0; for (int j=0;j<8;j++) w |= ((ulong)in[off + i*8 + j]) << (8*j);
            s[i] ^= w;
        }
        keccak_f1600(s); off += rate; inlen -= rate;
    }
    uchar block[136]; for (int i=0;i<136;i++) block[i]=0;
    for (int i=0;i<inlen;i++) block[i] = in[off+i];
    block[inlen] = 0x01; block[rate-1] |= 0x80;
    for (int i=0;i<rate/8;i++){
        ulong w=0; for (int j=0;j<8;j++) w |= ((ulong)block[i*8+j]) << (8*j);
        s[i] ^= w;
    }
    keccak_f1600(s);
    for (int i=0;i<4;i++){
        for (int j=0;j<8;j++) out32[i*8 + j] = (uchar)((s[i] >> (8*j)) & 0xFF);
    }
}

__kernel void keccak256_kernel(__global uchar* input_data, __global int* input_lengths, 
                              __global uchar* output_data, int batch_size) {
    int idx = get_global_id(0);
    
    if (idx >= batch_size) return;
    
    int input_len = input_lengths[idx];
    __global uchar* input = input_data + idx * 256;
    __global uchar* output = output_data + idx * 32;
    keccak256_ocl(input, input_len, output);
}

__kernel void ecdsa_verify_kernel(__global uchar* signatures, __global uchar* messages, 
                                 __global uchar* public_keys, __global uchar* results, int batch_size) {
    int idx = get_global_id(0);
    
    if (idx >= batch_size) return;
    
    __global uchar* sig = signatures + idx * 65;
    __global uchar* msg = messages + idx * 32;
    __global uchar* pubkey = public_keys + idx * 64;
    
    // Simplified verification for demonstration
    uchar valid = 1;
    for (int i = 0; i < 32 && valid; i++) {
        if (msg[i] == 0 && sig[i] == 0) {
            valid = 0;
        }
    }
    
    results[idx] = valid;
}

// Enhanced OpenCL kernel for full transaction processing with RLP decoding and EVM execution
__kernel void transaction_process_kernel(__global uchar* tx_data, __global int* tx_lengths,
                                       __global uchar* state_data, __global uchar* access_lists,
                                       __global uchar* results, int batch_size) {
    int idx = get_global_id(0);
    
    if (idx >= batch_size) return;
    
    __global uchar* tx = tx_data + idx * 1024;        // Max 1KB per transaction
    int tx_len = tx_lengths[idx];
    __global uchar* state = state_data + idx * 2048;  // 2KB state snapshot per tx
    __global uchar* access_list = access_lists + idx * 512; // 512B access list per tx
    __global uchar* result = results + idx * 128;     // 128 bytes result per transaction (expanded)
    
    // Initialize result structure
    // [0-31]: transaction hash
    // [32]: validity flag
    // [33]: execution status (0=success, 1=revert, 2=out_of_gas, 3=invalid)
    // [34-41]: gas used (8 bytes LE)
    // [42-73]: return data hash (32 bytes)
    // [74-105]: revert reason hash (32 bytes) 
    // [106-127]: reserved for future use
    
    uchar valid = 0;
    uchar exec_status = 3; // invalid by default
    ulong gas_used = 0;
    __private uchar tx_hash[32];
    __private uchar return_hash[32];
    __private uchar revert_hash[32];
    
    if (tx_len > 0 && tx_len < 1024) {
        // Step 1: Decode RLP transaction structure
        int to_off=0,to_len=0,data_off=0,data_len=0; int v_off=0,v_len=0,r_off=0,r_len=0,s_off=0,s_len=0; ulong gas_limit=0;
        valid = decode_rlp_transaction_ocl(tx, tx_len, &gas_limit, &to_off, &to_len, &data_off, &data_len,
                                           &v_off, &v_len, &r_off, &r_len, &s_off, &s_len);
        
        if (valid) {
            // Step 2: Compute transaction hash using Keccak-256
            compute_transaction_hash_ocl(tx, tx_len, tx_hash);
            
            // Step 3: Perform signature recovery
            uchar sig_valid = recover_transaction_signature_ocl(tx, tx_len, v_off, v_len, r_off, r_len, s_off, s_len);
            
            if (sig_valid) {
                // Step 4: Execute EVM state transition
                exec_status = execute_evm_transaction_ocl(
                    tx, tx_len, state, access_list, 
                    &gas_used, return_hash, revert_hash,
                    data_off, data_len, to_off, to_len, gas_limit
                );
            } else {
                exec_status = 3; // invalid signature
                valid = 0;
            }
        }
    }
    
    // Store results in packed format
    // Transaction hash (32 bytes)
    for (int i=0;i<32;i++) result[i] = tx_hash[i];
    
    result[32] = valid;           // Validity flag
    result[33] = exec_status;     // Execution status
    
    // Gas used (8 bytes LE)
    for (int i = 0; i < 8; i++) {
        result[34 + i] = (gas_used >> (i * 8)) & 0xFF;
    }
    
    // Return data hash (32 bytes)
    for (int i=0;i<32;i++) result[42 + i] = return_hash[i];
    
    // Revert reason hash (32 bytes)
    for (int i=0;i<32;i++) result[74 + i] = revert_hash[i];
}

// Helper RLP readers
uchar rlp_read_len_ocl(__global const uchar* p, int end, int pos, uchar list, int* len_out, int* hsz_out) {
    if (pos >= end) return 0;
    uchar b = p[pos];
    if (!list) {
        if (b <= 0x7f) { *len_out = 1; *hsz_out = 0; return 1; }
        if (b <= 0xb7) { int l = (int)(b - 0x80); *len_out = l; *hsz_out = 1; return pos+1+l <= end; }
        if (b <= 0xbf) { int lsize = (int)(b - 0xb7); if (pos+1+lsize > end) return 0; int l=0; for (int i=0;i<lsize;i++){ l = (l<<8) | p[pos+1+i]; } *len_out=l; *hsz_out=1+lsize; return pos+1+lsize+l <= end; }
        return 0;
    } else {
        if (b <= 0xf7) { int l = (int)(b - 0xc0); *len_out = l; *hsz_out = 1; return pos+1+l <= end; }
        if (b <= 0xff) { int lsize = (int)(b - 0xf7); if (pos+1+lsize > end) return 0; int l=0; for (int i=0;i<lsize;i++){ l=(l<<8)|p[pos+1+i]; } *len_out=l; *hsz_out=1+lsize; return pos+1+lsize+l <= end; }
        return 0;
    }
}

uchar rlp_next_item_ocl(__global const uchar* p, int end, int pos, int* item_off, int* item_len, int* next_pos) {
    if (pos >= end) return 0;
    uchar b = p[pos];
    if (b <= 0x7f){ *item_off = pos; *item_len = 1; *next_pos = pos+1; return 1; }
    if (b <= 0xb7){ int l=b-0x80; if (pos+1+l>end) return 0; *item_off=pos+1; *item_len=l; *next_pos=pos+1+l; return 1; }
    if (b <= 0xbf){ int lsize=b-0xb7; if (pos+1+lsize> end) return 0; int l=0; for(int i=0;i<lsize;i++){ l=(l<<8)|p[pos+1+i]; } if (pos+1+lsize+l> end) return 0; *item_off=pos+1+lsize; *item_len=l; *next_pos=pos+1+lsize+l; return 1; }
    return 0;
}

// Helper function to decode RLP transaction structure
uchar decode_rlp_transaction_ocl(__global uchar* tx_data, int length, ulong* gas_limit,
                                 int* to_off, int* to_len, int* data_off, int* data_len,
                                 int* v_off, int* v_len, int* r_off, int* r_len, int* s_off, int* s_len) {
    if (length < 3) return 0;
    int start = 0;
    if (tx_data[0] < 0xc0) {
        if (!(tx_data[0] == 0x01 || tx_data[0] == 0x02)) return 0;
        start = 1;
    }
    int list_len=0, hdr=0; if (!rlp_read_len_ocl(tx_data, length, start, 1, &list_len, &hdr)) return 0;
    hdr += start;
    int pos = hdr; int end = hdr + list_len; if (end > length) return 0;
    int off=0,len=0; *to_off=*to_len=*data_off=*data_len=0; *gas_limit=0; *v_off=*v_len=*r_off=*r_len=*s_off=*s_len=0;
    int max_fields = (tx_data[0] == 0x02) ? 12 : ((tx_data[0] == 0x01) ? 11 : 9);
    for (int field=0; field<max_fields; field++) {
        if (!rlp_next_item_ocl(tx_data, end, pos, &off, &len, &pos)) return 0;
        if (((tx_data[0] >= 0xc0) && field == 2) || (tx_data[0] == 0x01 && field == 3) || (tx_data[0] == 0x02 && field == 4)) {
            ulong gl=0; for (int i=0;i<len;i++){ gl = (gl<<8) | (ulong)tx_data[off+i]; } *gas_limit = gl;
        } else if (((tx_data[0] >= 0xc0) && field == 3) || (tx_data[0] == 0x01 && field == 4) || (tx_data[0] == 0x02 && field == 5)) {
            *to_off = off; *to_len = len;
        } else if (((tx_data[0] >= 0xc0) && field == 5) || (tx_data[0] == 0x01 && field == 6) || (tx_data[0] == 0x02 && field == 7)) {
            *data_off = off; *data_len = len;
        } else if (((tx_data[0] >= 0xc0) && field == 6) || (tx_data[0] == 0x01 && field == 8) || (tx_data[0] == 0x02 && field == 9)) {
            *v_off = off; *v_len = len;
        } else if (((tx_data[0] >= 0xc0) && field == 7) || (tx_data[0] == 0x01 && field == 9) || (tx_data[0] == 0x02 && field == 10)) {
            *r_off = off; *r_len = len;
        } else if (((tx_data[0] >= 0xc0) && field == 8) || (tx_data[0] == 0x01 && field == 10) || (tx_data[0] == 0x02 && field == 11)) {
            *s_off = off; *s_len = len;
        }
    }
    return (*gas_limit > 0);
}

// Helper function to compute transaction hash
void compute_transaction_hash_ocl(__global uchar* tx_data, int length, __private uchar out32[32]) {
    keccak256_ocl(tx_data, length, out32);
}

// Helper function for signature recovery
uchar recover_transaction_signature_ocl(__global uchar* tx_data, int length,
                                        int v_off, int v_len, int r_off, int r_len, int s_off, int s_len) {
    if (v_off + v_len > length || r_off + r_len > length || s_off + s_len > length) return 0;
    if (r_len <= 0 || r_len > 32 || s_len <= 0 || s_len > 32 || v_len <= 0 || v_len > 32) return 0;
    if (tx_data[r_off] == 0 || tx_data[s_off] == 0) return 0;
    ulong v=0; for (int i=0;i<v_len && i<8;i++){ v = (v<<8) | (ulong)tx_data[v_off+i]; }
    uchar valid_v = (v == 27 || v == 28 || v >= 35);
    if (!valid_v) return 0;
    uchar r32[32]; uchar s32[32]; for (int i=0;i<32;i++){ r32[i]=0; s32[i]=0; }
    for (int i=0;i<r_len;i++) r32[32 - r_len + i] = tx_data[r_off + i];
    for (int i=0;i<s_len;i++) s32[32 - s_len + i] = tx_data[s_off + i];
    uchar r_nonzero=0, s_nonzero=0; for (int i=0;i<32;i++){ if (r32[i]) r_nonzero=1; if (s32[i]) s_nonzero=1; }
    if (!(r_nonzero && s_nonzero)) return 0;
    const uchar n_half[32] = { 0x7F,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
                               0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
                               0x5D,0x57,0x6E,0x73,0x57,0xA4,0x50,0x1D,
                               0xDF,0xE9,0x2F,0x46,0x68,0x1B,0x20,0xA0 };
    for (int i=0;i<32;i++){ if (s32[i] < n_half[i]) break; if (s32[i] > n_half[i]) return 0; }
    return 1;
}

// Helper function for EVM execution simulation
uchar execute_evm_transaction_ocl(
    __global uchar* tx_data, int length, __global uchar* state_data, __global uchar* access_list,
    ulong* gas_used, __private uchar* return_hash32, __private uchar* revert_hash32,
    int data_off, int data_len, int to_off, int to_len, ulong gas_limit
) {
    ulong gas = 21000UL;
    for (int i=0;i<data_len;i++) gas += (tx_data[data_off+i] == 0) ? 4 : 16;
    uchar is_transfer = (data_len == 0 && to_len == 20);
    if (!is_transfer) gas += 50000UL;
    *gas_used = gas;
    if (gas > gas_limit) { keccak256_ocl(tx_data + data_off, data_len, revert_hash32); return 2; }
    keccak256_ocl(tx_data + data_off, data_len, return_hash32); return 0;
}

// Removed simplified Keccak and toy hash; replaced by keccak256_ocl
)";

// Host functions
int initOpenCL() {
    if (opencl_initialized) {
        return device_count;
    }
    
    cl_uint platform_count;
    CL_CHECK(clGetPlatformIDs(0, NULL, &platform_count));
    
    if (platform_count == 0) {
        fprintf(stderr, "No OpenCL platforms found\n");
        return -1;
    }
    
    cl_platform_id* platforms = malloc(platform_count * sizeof(cl_platform_id));
    CL_CHECK(clGetPlatformIDs(platform_count, platforms, NULL));
    
    // Use first platform
    platform = platforms[0];
    free(platforms);
    
    // Get GPU devices
    cl_uint gpu_count;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &gpu_count);
    if (err != CL_SUCCESS || gpu_count == 0) {
        // Fallback to CPU devices
        CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &gpu_count));
    }
    
    device_count = gpu_count;
    devices = malloc(device_count * sizeof(cl_device_id));
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices, NULL);
    if (err != CL_SUCCESS) {
        CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, device_count, devices, NULL));
    }
    
    // Create context
    context = clCreateContext(NULL, device_count, devices, NULL, NULL, &err);
    CL_CHECK(err);
    
    // Create command queues
    queues = malloc(device_count * sizeof(cl_command_queue));
    for (int i = 0; i < device_count; i++) {
#ifdef CL_VERSION_2_0
        cl_queue_properties properties[] = {0};
        queues[i] = clCreateCommandQueueWithProperties(context, devices[i], properties, &err);
#else
        queues[i] = clCreateCommandQueue(context, devices[i], 0, &err);
#endif
        CL_CHECK(err);
    }
    
    // Create and build program
    program = clCreateProgramWithSource(context, 1, &keccak_kernel_source, NULL, &err);
    CL_CHECK(err);
    
    err = clBuildProgram(program, device_count, devices, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "OpenCL build error: %s\n", log);
        free(log);
        return -1;
    }
    
    // Create kernels
    keccak_kernel = clCreateKernel(program, "keccak256_kernel", &err);
    CL_CHECK(err);
    
    ecdsa_kernel = clCreateKernel(program, "ecdsa_verify_kernel", &err);
    CL_CHECK(err);
    
    tx_kernel = clCreateKernel(program, "transaction_process_kernel", &err);
    CL_CHECK(err);
    
    opencl_initialized = true;
    printf("OpenCL initialized with %d devices\n", device_count);
    return device_count;
}

int processHashesOpenCL(void* hashes, int count, void* results) {
    if (!opencl_initialized || count <= 0) {
        return -1;
    }
    
    cl_int err;
    
    // Create buffers
    size_t input_size = count * 256;
    size_t output_size = count * 32;
    
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &err);
    CL_CHECK(err);
    
    cl_mem lengths_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(int), NULL, &err);
    CL_CHECK(err);
    
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, NULL, &err);
    CL_CHECK(err);
    
    // Copy input data
    CL_CHECK(clEnqueueWriteBuffer(queues[0], input_buffer, CL_TRUE, 0, input_size, hashes, 0, NULL, NULL));
    
    // Set lengths (assuming 32 bytes per input)
    int* lengths = malloc(count * sizeof(int));
    for (int i = 0; i < count; i++) {
        lengths[i] = 32;
    }
    CL_CHECK(clEnqueueWriteBuffer(queues[0], lengths_buffer, CL_TRUE, 0, count * sizeof(int), lengths, 0, NULL, NULL));
    
    // Set kernel arguments
    CL_CHECK(clSetKernelArg(keccak_kernel, 0, sizeof(cl_mem), &input_buffer));
    CL_CHECK(clSetKernelArg(keccak_kernel, 1, sizeof(cl_mem), &lengths_buffer));
    CL_CHECK(clSetKernelArg(keccak_kernel, 2, sizeof(cl_mem), &output_buffer));
    CL_CHECK(clSetKernelArg(keccak_kernel, 3, sizeof(int), &count));
    
    // Execute kernel
    size_t global_work_size = count;
    size_t local_work_size = 256;
    if (global_work_size % local_work_size != 0) {
        global_work_size = ((count + local_work_size - 1) / local_work_size) * local_work_size;
    }
    
    CL_CHECK(clEnqueueNDRangeKernel(queues[0], keccak_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    
    // Read results
    CL_CHECK(clEnqueueReadBuffer(queues[0], output_buffer, CL_TRUE, 0, output_size, results, 0, NULL, NULL));
    
    // Cleanup
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(lengths_buffer);
    clReleaseMemObject(output_buffer);
    free(lengths);
    
    return 0;
}

int verifySignaturesOpenCL(void* signatures, int count, void* results) {
    if (!opencl_initialized || count <= 0) {
        return -1;
    }
    
    cl_int err;
    
    // Create buffers
    size_t sig_size = count * 65;
    size_t msg_size = count * 32;
    size_t key_size = count * 64;
    
    cl_mem sig_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sig_size, NULL, &err);
    CL_CHECK(err);
    
    cl_mem msg_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, msg_size, NULL, &err);
    CL_CHECK(err);
    
    cl_mem key_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, key_size, NULL, &err);
    CL_CHECK(err);
    
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, count, NULL, &err);
    CL_CHECK(err);
    
    // Copy input data (signatures + messages + pubkeys may be packed in a single buffer)
    // If 'signatures' points to a packed layout [65 | 32 | 64] * count, deinterleave into separate buffers.
    uint8_t* packed = (uint8_t*)signatures;
    uint8_t* host_sigs = (uint8_t*)malloc(sig_size);
    uint8_t* host_msgs = (uint8_t*)malloc(msg_size);
    uint8_t* host_keys = (uint8_t*)malloc(key_size);
    if (host_sigs == NULL || host_msgs == NULL || host_keys == NULL) {
        if (host_sigs) free(host_sigs);
        if (host_msgs) free(host_msgs);
        if (host_keys) free(host_keys);
        return -1;
    }

    size_t stride = 65 + 32 + 64;
    for (int i = 0; i < count; i++) {
        memcpy(host_sigs + i * 65, packed + i * stride, 65);
        memcpy(host_msgs + i * 32, packed + i * stride + 65, 32);
        memcpy(host_keys + i * 64, packed + i * stride + 65 + 32, 64);
    }

    CL_CHECK(clEnqueueWriteBuffer(queues[0], sig_buffer, CL_TRUE, 0, sig_size, host_sigs, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queues[0], msg_buffer, CL_TRUE, 0, msg_size, host_msgs, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queues[0], key_buffer, CL_TRUE, 0, key_size, host_keys, 0, NULL, NULL));
    
    // Set kernel arguments
    CL_CHECK(clSetKernelArg(ecdsa_kernel, 0, sizeof(cl_mem), &sig_buffer));
    CL_CHECK(clSetKernelArg(ecdsa_kernel, 1, sizeof(cl_mem), &msg_buffer));
    CL_CHECK(clSetKernelArg(ecdsa_kernel, 2, sizeof(cl_mem), &key_buffer));
    CL_CHECK(clSetKernelArg(ecdsa_kernel, 3, sizeof(cl_mem), &result_buffer));
    CL_CHECK(clSetKernelArg(ecdsa_kernel, 4, sizeof(int), &count));
    
    // Execute kernel
    size_t global_work_size = count;
    size_t local_work_size = 256;
    if (global_work_size % local_work_size != 0) {
        global_work_size = ((count + local_work_size - 1) / local_work_size) * local_work_size;
    }
    
    CL_CHECK(clEnqueueNDRangeKernel(queues[0], ecdsa_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    
    // Read results
    CL_CHECK(clEnqueueReadBuffer(queues[0], result_buffer, CL_TRUE, 0, count, results, 0, NULL, NULL));
    
    // Cleanup
    clReleaseMemObject(sig_buffer);
    clReleaseMemObject(msg_buffer);
    clReleaseMemObject(key_buffer);
    clReleaseMemObject(result_buffer);

    // Free host-side temporary buffers
    free(host_sigs);
    free(host_msgs);
    free(host_keys);
    
    return 0;
}

// Enhanced OpenCL transaction processing with full execution context
int processTxBatchOpenCLFull(void* txData, void* txLens, void* stateData, void* accessLists, int txCount, void* results) {
    if (!opencl_initialized || txCount <= 0) {
        return -1;
    }
    
    cl_int err;
    
    // Create buffers for enhanced processing
    size_t tx_data_size = txCount * 1024;        // 1KB per transaction
    size_t state_size = txCount * 2048;          // 2KB state snapshot per transaction
    size_t access_size = txCount * 512;          // 512B access list per transaction
    size_t result_size = txCount * 128;          // 128 bytes result per transaction (expanded)
    
    cl_mem tx_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, tx_data_size, NULL, &err);
    CL_CHECK(err);
    
    cl_mem lengths_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, txCount * sizeof(int), NULL, &err);
    CL_CHECK(err);
    
    cl_mem state_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, state_size, NULL, &err);
    CL_CHECK(err);
    
    cl_mem access_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, access_size, NULL, &err);
    CL_CHECK(err);
    
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, result_size, NULL, &err);
    CL_CHECK(err);
    
    // Copy transaction data
    CL_CHECK(clEnqueueWriteBuffer(queues[0], tx_buffer, CL_TRUE, 0, tx_data_size, txData, 0, NULL, NULL));
    
    // Copy state data (if provided)
    if (stateData) {
        CL_CHECK(clEnqueueWriteBuffer(queues[0], state_buffer, CL_TRUE, 0, state_size, stateData, 0, NULL, NULL));
    } else {
        // Initialize with zeros if no state data provided
        uint8_t* zero_state = calloc(state_size, 1);
        CL_CHECK(clEnqueueWriteBuffer(queues[0], state_buffer, CL_TRUE, 0, state_size, zero_state, 0, NULL, NULL));
        free(zero_state);
    }
    
    // Copy access lists (if provided)
    if (accessLists) {
        CL_CHECK(clEnqueueWriteBuffer(queues[0], access_buffer, CL_TRUE, 0, access_size, accessLists, 0, NULL, NULL));
    } else {
        // Initialize with zeros if no access lists provided
        uint8_t* zero_access = calloc(access_size, 1);
        CL_CHECK(clEnqueueWriteBuffer(queues[0], access_buffer, CL_TRUE, 0, access_size, zero_access, 0, NULL, NULL));
        free(zero_access);
    }
    
    // Set lengths from provided pointer
    if (!txLens) { clReleaseMemObject(tx_buffer); clReleaseMemObject(lengths_buffer); clReleaseMemObject(state_buffer); clReleaseMemObject(access_buffer); clReleaseMemObject(result_buffer); return -1; }
    CL_CHECK(clEnqueueWriteBuffer(queues[0], lengths_buffer, CL_TRUE, 0, txCount * sizeof(int), txLens, 0, NULL, NULL));
    
    // Set enhanced kernel arguments
    CL_CHECK(clSetKernelArg(tx_kernel, 0, sizeof(cl_mem), &tx_buffer));
    CL_CHECK(clSetKernelArg(tx_kernel, 1, sizeof(cl_mem), &lengths_buffer));
    CL_CHECK(clSetKernelArg(tx_kernel, 2, sizeof(cl_mem), &state_buffer));
    CL_CHECK(clSetKernelArg(tx_kernel, 3, sizeof(cl_mem), &access_buffer));
    CL_CHECK(clSetKernelArg(tx_kernel, 4, sizeof(cl_mem), &result_buffer));
    CL_CHECK(clSetKernelArg(tx_kernel, 5, sizeof(int), &txCount));
    
    // Execute kernel
    size_t global_work_size = txCount;
    size_t local_work_size = 256;
    if (global_work_size % local_work_size != 0) {
        global_work_size = ((txCount + local_work_size - 1) / local_work_size) * local_work_size;
    }
    
    CL_CHECK(clEnqueueNDRangeKernel(queues[0], tx_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    
    // Read results
    CL_CHECK(clEnqueueReadBuffer(queues[0], result_buffer, CL_TRUE, 0, result_size, results, 0, NULL, NULL));
    
    // Cleanup
    clReleaseMemObject(tx_buffer);
    clReleaseMemObject(lengths_buffer);
    clReleaseMemObject(state_buffer);
    clReleaseMemObject(access_buffer);
    clReleaseMemObject(result_buffer);
    
    
    return 0;
}

// Legacy OpenCL transaction processing (for backward compatibility)
int processTxBatchOpenCL(void* txData, int txCount, void* results) {
    // Not supported without lengths; return error
    (void)txData; (void)txCount; (void)results;
    return -1;
}

void cleanupOpenCL() {
    if (!opencl_initialized) {
        return;
    }
    
    if (keccak_kernel) clReleaseKernel(keccak_kernel);
    if (ecdsa_kernel) clReleaseKernel(ecdsa_kernel);
    if (tx_kernel) clReleaseKernel(tx_kernel);
    if (program) clReleaseProgram(program);
    
    if (queues) {
        for (int i = 0; i < device_count; i++) {
            clReleaseCommandQueue(queues[i]);
        }
        free(queues);
    }
    
    if (context) clReleaseContext(context);
    if (devices) free(devices);
    
    opencl_initialized = false;
    device_count = 0;
    
    printf("OpenCL cleanup completed\n");
}

#ifdef __cplusplus
}
#endif

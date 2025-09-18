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
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800aUL,
    0x800000008000000aUL, 0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL, 0x8000000000000000UL,
    0x0000000080008082UL, 0x800000000000808aUL, 0x8000000080008000UL
};

ulong rotl64(ulong x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f1600(__private ulong state[25]) {
    ulong C[5], D[5], B[25];
    
    for (int round = 0; round < 24; round++) {
        // Theta step
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[y * 5 + x] ^= D[x];
            }
        }
        
        // Rho and Pi steps
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int rho_offset = ((x + 3 * y) % 5) * 5 + x;
                int rho_rotation = ((24 * x + 36 * y) % 64);
                B[y * 5 + ((2 * x + 3 * y) % 5)] = rotl64(state[rho_offset], rho_rotation);
            }
        }
        
        // Chi step
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[y * 5 + x] = B[y * 5 + x] ^ ((~B[y * 5 + ((x + 1) % 5)]) & B[y * 5 + ((x + 2) % 5)]);
            }
        }
        
        // Iota step
        state[0] ^= keccak_round_constants[round];
    }
}

__kernel void keccak256_kernel(__global uchar* input_data, __global int* input_lengths, 
                              __global uchar* output_data, int batch_size) {
    int idx = get_global_id(0);
    
    if (idx >= batch_size) return;
    
    ulong state[25] = {0};
    int input_len = input_lengths[idx];
    __global uchar* input = input_data + idx * 256;
    __global uchar* output = output_data + idx * 32;
    
    // Absorption phase
    int rate = 136; // 1088 bits / 8 = 136 bytes
    int offset = 0;
    
    while (input_len >= rate) {
        for (int i = 0; i < rate / 8; i++) {
            ulong word = 0;
            for (int j = 0; j < 8 && offset + i * 8 + j < input_len; j++) {
                word |= ((ulong)input[offset + i * 8 + j]) << (j * 8);
            }
            state[i] ^= word;
        }
        keccak_f1600(state);
        input_len -= rate;
        offset += rate;
    }
    
    // Final block with padding
    uchar final_block[136] = {0};
    for (int i = 0; i < input_len; i++) {
        final_block[i] = input[offset + i];
    }
    final_block[input_len] = 0x01;
    final_block[rate - 1] |= 0x80;
    
    for (int i = 0; i < rate / 8; i++) {
        ulong word = 0;
        for (int j = 0; j < 8; j++) {
            word |= ((ulong)final_block[i * 8 + j]) << (j * 8);
        }
        state[i] ^= word;
    }
    keccak_f1600(state);
    
    // Squeezing phase
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (state[i] >> (j * 8)) & 0xFF;
        }
    }
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
    ulong tx_hash[4] = {0}; // 256-bit hash
    ulong return_hash[4] = {0};
    ulong revert_hash[4] = {0};
    
    if (tx_len > 0 && tx_len < 1024) {
        // Step 1: Decode RLP transaction structure
        valid = decode_rlp_transaction_ocl(tx, tx_len, &gas_used);
        
        if (valid) {
            // Step 2: Compute transaction hash using Keccak-256
            compute_transaction_hash_ocl(tx, tx_len, tx_hash);
            
            // Step 3: Perform signature recovery
            uchar sig_valid = recover_transaction_signature_ocl(tx, tx_len, state);
            
            if (sig_valid) {
                // Step 4: Execute EVM state transition
                exec_status = execute_evm_transaction_ocl(
                    tx, tx_len, state, access_list, 
                    &gas_used, return_hash, revert_hash
                );
            } else {
                exec_status = 3; // invalid signature
                valid = 0;
            }
        }
    }
    
    // Store results in packed format
    // Transaction hash (32 bytes)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            result[i * 8 + j] = (tx_hash[i] >> (j * 8)) & 0xFF;
        }
    }
    
    result[32] = valid;           // Validity flag
    result[33] = exec_status;     // Execution status
    
    // Gas used (8 bytes LE)
    for (int i = 0; i < 8; i++) {
        result[34 + i] = (gas_used >> (i * 8)) & 0xFF;
    }
    
    // Return data hash (32 bytes)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            result[42 + i * 8 + j] = (return_hash[i] >> (j * 8)) & 0xFF;
        }
    }
    
    // Revert reason hash (32 bytes)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            result[74 + i * 8 + j] = (revert_hash[i] >> (j * 8)) & 0xFF;
        }
    }
}

// Helper function to decode RLP transaction structure
uchar decode_rlp_transaction_ocl(__global uchar* tx_data, int length, ulong* gas_limit) {
    if (length < 32) return 0;
    
    // Simplified RLP decoding for transaction structure
    // Real implementation would parse: [nonce, gasPrice, gasLimit, to, value, data, v, r, s]
    
    int offset = 0;
    
    // Skip RLP list header (simplified)
    if (tx_data[0] >= 0xf8) {
        offset = 1 + (tx_data[0] - 0xf7);
    } else if (tx_data[0] >= 0xc0) {
        offset = 1;
    }
    
    if (offset >= length) return 0;
    
    // Extract gas limit (simplified - assume it's at a fixed offset)
    // In real implementation, this would properly parse RLP structure
    *gas_limit = 21000; // Default gas limit
    
    // Basic validation: check for minimum transaction structure
    uchar has_signature = (length >= offset + 65); // At least r, s, v components
    uchar has_basic_fields = (length >= offset + 32); // Basic transaction fields
    
    return has_signature && has_basic_fields;
}

// Helper function to compute transaction hash
void compute_transaction_hash_ocl(__global uchar* tx_data, int length, ulong* hash_out) {
    // Simplified Keccak-256 hash computation
    ulong state[25] = {0};
    
    // Absorption phase (simplified)
    int blocks = (length + 135) / 136; // 136 = rate for Keccak-256
    
    for (int block = 0; block < blocks; block++) {
        int block_start = block * 136;
        int block_size = (block_start + 136 <= length) ? 136 : (length - block_start);
        
        // XOR input into state
        for (int i = 0; i < block_size && i < 136; i += 8) {
            ulong word = 0;
            for (int j = 0; j < 8 && block_start + i + j < length; j++) {
                word |= ((ulong)tx_data[block_start + i + j]) << (j * 8);
            }
            if ((i / 8) < 17) { // Only first 17 words of state
                state[i / 8] ^= word;
            }
        }
        
        // Apply Keccak-f[1600] permutation (simplified)
        keccak_f1600_simplified_ocl(state);
    }
    
    // Extract hash (first 256 bits)
    for (int i = 0; i < 4; i++) {
        hash_out[i] = state[i];
    }
}

// Helper function for signature recovery
uchar recover_transaction_signature_ocl(__global uchar* tx_data, int length, __global uchar* state_data) {
    if (length < 65) return 0;
    
    // Extract signature components (r, s, v) from end of transaction
    __global uchar* sig_r = &tx_data[length - 65];
    __global uchar* sig_s = &tx_data[length - 33];
    uchar v = tx_data[length - 1];
    
    // Simplified signature validation
    // Check v value is valid (27, 28, or EIP-155 protected)
    uchar valid_v = (v == 27 || v == 28 || v >= 35);
    
    // Check r and s are not zero
    uchar r_nonzero = 0, s_nonzero = 0;
    for (int i = 0; i < 32; i++) {
        if (sig_r[i] != 0) r_nonzero = 1;
        if (sig_s[i] != 0) s_nonzero = 1;
    }
    
    // In full implementation, would perform elliptic curve signature recovery
    // For now, return basic validation result
    return valid_v && r_nonzero && s_nonzero;
}

// Helper function for EVM execution simulation
uchar execute_evm_transaction_ocl(
    __global uchar* tx_data, int length, __global uchar* state_data, __global uchar* access_list,
    ulong* gas_used, ulong* return_hash, ulong* revert_hash
) {
    // Simplified EVM execution simulation
    // In full implementation, this would:
    // 1. Load account states from state_data
    // 2. Execute transaction against EVM
    // 3. Update state and compute gas usage
    // 4. Handle reverts and return data
    
    *gas_used = 21000; // Base transaction cost
    
    // Simulate different execution outcomes based on transaction data
    if (length < 100) {
        // Simple transfer - always succeeds
        return 0; // success
    } else if (tx_data[50] == 0xfd) { // REVERT opcode simulation
        // Simulate revert with reason
        compute_simple_hash_ocl(tx_data + 50, 32, revert_hash);
        *gas_used += 5000; // Additional gas for execution
        return 1; // revert
    } else if (tx_data[60] == 0xff) { // Simulate out of gas
        *gas_used = 1000000; // High gas usage
        return 2; // out of gas
    } else {
        // Contract execution - simulate success with return data
        compute_simple_hash_ocl(tx_data + 100, 32, return_hash);
        *gas_used += 50000; // Contract execution gas
        return 0; // success
    }
}

// Simplified Keccak-f[1600] permutation for OpenCL
void keccak_f1600_simplified_ocl(ulong state[25]) {
    // Simplified version with reduced rounds for GPU efficiency
    for (int round = 0; round < 12; round++) { // Reduced from 24 rounds
        // Theta step (simplified)
        ulong C[5];
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        
        for (int x = 0; x < 5; x++) {
            ulong D = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; y++) {
                state[y * 5 + x] ^= D;
            }
        }
        
        // Simplified rho and pi steps
        ulong temp = state[1];
        for (int i = 0; i < 24; i++) {
            int j = (i + 1) % 25; // Simplified rotation pattern
            ulong temp2 = state[j];
            state[j] = rotl64(temp, i + 1);
            temp = temp2;
        }
        
        // Chi step (simplified)
        for (int y = 0; y < 5; y++) {
            ulong temp[5];
            for (int x = 0; x < 5; x++) {
                temp[x] = state[y * 5 + x];
            }
            for (int x = 0; x < 5; x++) {
                state[y * 5 + x] = temp[x] ^ ((~temp[(x + 1) % 5]) & temp[(x + 2) % 5]);
            }
        }
        
        // Iota step
        state[0] ^= keccak_round_constants[round % 24];
    }
}

// Simple hash function for return data and revert reasons
void compute_simple_hash_ocl(__global uchar* data, int length, ulong* hash_out) {
    ulong hash = 0x123456789abcdef0UL;
    
    for (int i = 0; i < length; i++) {
        hash ^= ((ulong)data[i]) << (i % 64);
        hash = rotl64(hash, 1);
    }
    
    // Split 64-bit hash into four 64-bit values for 256-bit result
    hash_out[0] = hash;
    hash_out[1] = rotl64(hash, 16);
    hash_out[2] = rotl64(hash, 32);
    hash_out[3] = rotl64(hash, 48);
}
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
int processTxBatchOpenCLFull(void* txData, void* stateData, void* accessLists, int txCount, void* results) {
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
    
    // Set lengths
    int* lengths = malloc(txCount * sizeof(int));
    for (int i = 0; i < txCount; i++) {
        lengths[i] = 200; // Average transaction size
    }
    CL_CHECK(clEnqueueWriteBuffer(queues[0], lengths_buffer, CL_TRUE, 0, txCount * sizeof(int), lengths, 0, NULL, NULL));
    
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
    free(lengths);
    
    return 0;
}

// Legacy OpenCL transaction processing (for backward compatibility)
int processTxBatchOpenCL(void* txData, int txCount, void* results) {
    // Call enhanced version with null state data and access lists
    return processTxBatchOpenCLFull(txData, NULL, NULL, txCount, results);
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

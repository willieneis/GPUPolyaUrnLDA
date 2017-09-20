#pragma once

#include "types.cuh"
#include "tuning.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#include <cstdio>
#include "assert.h"

#define GPLDA_HASH_EMPTY 0xfffff // 20 bits
#define GPLDA_HASH_LINE_SIZE 16
#define GPLDA_HASH_MAX_NUM_LINES 4
#define GPLDA_HASH_NULL_POINTER 0x7f // 7 bits

namespace gplda {

union HashMapEntry {
  struct {
    u32 relocate: 1;
    u32 pointer: 7;
    u32 key: 20;
    u64 value: 36;
  };
  u64 int_repr;
};

// #pragma pack is unsupported in device code, so we can't force the bit field to be correct, but we can check for correctness using sizeof, guaranteed identical on host and device
static_assert(sizeof(HashMapEntry) == sizeof(u64), "bit field struct not properly packed in HashMapEntry");

template<SynchronizationType sync_type>
struct HashMap {
  u32 size;
  u32 max_size;
  HashMapEntry* data;
  HashMapEntry* temp_data;
  u32 a;
  u32 b;
  u32 c;
  u32 d;
  u32 needs_rebuild;
  curandStatePhilox4_32_10_t* rng;
  u32 ring_buffer_start;
  u32 ring_buffer_read_end;
  u32 ring_buffer_write_end;
  u32 ring_buffer_size;
  u32* ring_buffer_queue;
  HashMapEntry* ring_buffer;




  __device__ __forceinline__ void sync() {
    if(sync_type == block) {
      __syncthreads();
    }
  }





  __device__ __forceinline__ void ring_buffer_push(u32 element) {
    // determine constants
    u32 lane_idx = threadIdx.x % warpSize;

    // divergent threads may enter, so determine which thread will write
    u32 active_threads = __ballot(true);
    u32 leader = __ffs(active_threads) - 1;
    u32 warp_num_threads = __popc(active_threads);
    u32 offset = __popc((~(0xffffffff << lane_idx)) & active_threads);

    // increment queue size once per warp and broadcast to all lanes
    u32 warp_start;
    if(lane_idx == leader) {
      warp_start = atomicAdd(&ring_buffer_write_end, warp_num_threads);
    }
    warp_start = __shfl(warp_start, leader);

    // write elements to queue
    ring_buffer_queue[(warp_start + offset) % size] = element;

    // increment number of elements that may be read from queue
    if(lane_idx == leader) {
      do {} while(atomicCAS(&ring_buffer_read_end, warp_start, warp_start + warp_num_threads));
    }
  }

  __device__ __forceinline__ u32 ring_buffer_pop() {
    // determine constants
    u32 lane_idx = threadIdx.x % warpSize;

    // divergent threads may enter, so determine which thread will write
    u32 active_threads = __ballot(true);
    u32 leader = __ffs(active_threads) - 1;
    u32 warp_num_threads = __popc(active_threads);
    u32 offset = __popc((~(0xffffffff << lane_idx)) & active_threads);

    // read index from queue and broadcast
    u32 warp_start;
    if(lane_idx == leader) {
      warp_start = atomicAdd(&ring_buffer_start, warp_num_threads);
    }
    warp_start = __shfl(warp_start, leader);

    // return index of buffer location
    return warp_start + offset;
  }

  __device__ __forceinline__ void ring_buffer_init(HashMapEntry* b, u32* q, u32 s) {
    // calculate initialization variables common for all threads
    i32 dim = (sync_type == block) ? blockDim.x : warpSize;
    i32 thread_idx = threadIdx.x % dim;

    if(thread_idx == 0) {
      ring_buffer_start = 0;
      ring_buffer_read_end = 0;
      ring_buffer_write_end = 0;
      ring_buffer_size = s;
      ring_buffer_queue = q;
      ring_buffer = b;
    }

    // ensure parameter writes are visible to all threads
    sync();

    // set buffer to empty and queue to full
    for(i32 offset = 0; offset < ring_buffer_size / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < ring_buffer_size) {
        ring_buffer_queue[i] = i;
        ring_buffer[i] = entry(false,GPLDA_HASH_NULL_POINTER,GPLDA_HASH_EMPTY,0);
      }
    }

    // ensure queue writes are visible to all threads
    sync();
  }






  __device__ __forceinline__ i32 hash_slot(u32 key, i32 x, i32 y) {
    return (((x * key + y) % 334214459) % (size / GPLDA_HASH_LINE_SIZE)) * GPLDA_HASH_LINE_SIZE;
  }

  __device__ __forceinline__ i32 key_distance(u32 key, u32 slot) {
    u32 initial_slot = hash_slot(key,a,b);
    u32 stride = hash_slot(key,c,d);
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES - 1; ++i) {
      if((initial_slot + i*stride) % size == slot) {
        return i;
      }
    }
    return GPLDA_HASH_MAX_NUM_LINES;
  }


  __device__ __forceinline__ HashMapEntry entry(u64 ir) {
    HashMapEntry entry;
    entry.int_repr = ir;
    return entry;
  }

  __device__ __forceinline__ HashMapEntry entry(u32 r, u32 p, u32 k, u64 v) {
    HashMapEntry entry;
    entry.relocate = r;
    entry.pointer = p;
    entry.key = k;
    entry.value = v;
    return entry;
  }




  __device__ inline void init(void* in_data, u32 in_data_size, u32 initial_size, u32 num_concurrent_elements, curandStatePhilox4_32_10_t* in_rng) {
    // calculate initialization variables common for all threads
    i32 dim = (sync_type == block) ? blockDim.x : warpSize;
    i32 thread_idx = threadIdx.x % dim;

    // set map parameters and calculate random hash functions
    if(thread_idx == 0) {
      // round down to ensure cache alignment
      max_size = (((in_data_size - 3*num_concurrent_elements)/2) / GPLDA_HASH_LINE_SIZE) * GPLDA_HASH_LINE_SIZE;
      size = min((initial_size / GPLDA_HASH_LINE_SIZE + 1) * GPLDA_HASH_LINE_SIZE, max_size);

      // perform pointer arithmetic
      data = (HashMapEntry*) in_data;
      temp_data = data + max_size; // no sizeof for typed pointer arithmetic

      needs_rebuild = 0;
      rng = in_rng; // make sure this->rng is set before use
      float4 r = curand_uniform4(rng);
      a = __float2uint_rz(size * r.w);
      b = __float2uint_rz(size * r.x);
      c = __float2uint_rz(size * r.y);
      d = __float2uint_rz(size * r.z);
    }

    // synchronize to ensure shared memory writes are visible
    sync();

    // set map to empty
    for(i32 offset = 0; offset < size / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < size) {
        data[i] = entry(false,GPLDA_HASH_NULL_POINTER,GPLDA_HASH_EMPTY,0);
      }
    }

    ring_buffer_init(temp_data + max_size, (u32*) (temp_data + max_size + 2*num_concurrent_elements), 2*num_concurrent_elements);

    // synchronize to ensure initialization is complete
    sync();
  }





  __device__ inline void rebuild() {

  }





  __device__ inline u32 get2(u32 key) {
    // shuffle key to entire half-warp
    key = __shfl(key, 0, warpSize/2);
    i32 half_lane_idx = threadIdx.x % (warpSize / 2);
    u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / 16) * 4); // 4 if lane >= 16, 0 otherwise

    // check table
    i32 initial_slot = hash_slot(key,a,b);
    i32 stride = hash_slot(key,c,d);
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
      // compute slot and retrieve entry
      i32 slot = (initial_slot + i * stride) % size;
      HashMapEntry entry = data[slot + half_lane_idx];

      // check if we found the key, following pointers if necessary
      u32 key_found;
      u32 key_pointer;
      do {
        // check if we found key, return its value if so
        key_found = __ballot(entry.key == key) & half_lane_mask;
        if(key_found != 0) {
          return __shfl(entry.value, __ffs(key_found), warpSize/2);
        }

        // check if we found pointer, get its entry if so
        key_pointer = __ballot(entry.pointer != GPLDA_HASH_NULL_POINTER) & half_lane_mask;
        if(key_pointer != 0) {
          entry = ring_buffer[entry.pointer];
        }
      } while(key_pointer != 0);

      // check if Robin Hood guarantee indicates no key is present
      u32 no_key = __ballot(entry.key == GPLDA_HASH_EMPTY || key_distance(entry.key, slot) < i) & half_lane_mask;
      if(no_key != 0) {
        return 0;
      }
    }

    // ran out of possible slots: key not present
    return 0;
  }





  __device__ inline i32 try_accumulate2(u32 key, i32 diff) {
    // determine half warp indices
    i32 lane_idx = threadIdx.x % warpSize;
    i32 half_lane_idx = threadIdx.x % (warpSize / 2);
    u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / 16) * 4); // 4 if lane >= 16, 0 otherwise

    // build entry to be inserted and shuffle to entire half warp
    HashMapEntry half_warp_entry = entry(false,GPLDA_HASH_NULL_POINTER,key,diff);
    half_warp_entry.int_repr = __shfl(half_warp_entry.int_repr, 0, warpSize/2);

    // insert key into linked queue
    HashMapEntry thread_table_entry;
    i32 slot = hash_slot(key,a,b);
    i32 stride = hash_slot(key,c,d);
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
      // compute slot
      slot = (slot + stride) % size;

      // try to insert, retrying if race condition indicates it is necessary
      u32 retry;
      do {
        // retrieve entry for current half lane, set constants
        thread_table_entry = data[slot + half_lane_idx];
        retry = 0;

        // determine whether we found the key, an empty slot, or no key is present
        u32 thread_found_key = thread_table_entry.key == key;
        u32 thread_found_empty = thread_table_entry.key == GPLDA_HASH_EMPTY;
        u32 thread_no_key = key_distance(thread_table_entry.key, slot) > i;

        // determine which thread should write
        u32 half_warp_found_key = __ballot(thread_found_key) & half_lane_mask;
        u32 half_warp_found_empty = __ballot(thread_found_empty) & half_lane_mask;
        u32 half_warp_no_key = __ballot(thread_no_key) & half_lane_mask;

        u32 half_warp_write;
        if(half_warp_found_key != 0) {
          half_warp_write = half_warp_found_key;
        } else if(half_warp_found_empty != 0) {
          half_warp_write = half_warp_found_empty;
        } else if(half_warp_no_key != 0) {
          half_warp_write = half_warp_no_key;
        } else {
          half_warp_write = 0;
        }
        u32 lane_write_idx = __ffs(half_warp_write) - 1; // __ffs uses 1-based indexing

        if(half_warp_write != 0 && lane_idx == lane_write_idx) {
          // prepare new entry for table
          u32 buffer_idx = GPLDA_HASH_NULL_POINTER;

          // determine what kind of new entry we have
          if(thread_found_key == true) {
            // key found: accumulate value
            half_warp_entry.value += thread_table_entry.value;
          } else if(thread_no_key == true) {
            // Robin Hood guarantee indicates no key present: insert into eviction queue
            buffer_idx = ring_buffer_pop();
            ring_buffer[buffer_idx] = half_warp_entry;

            // prepare new entry
            half_warp_entry = thread_table_entry;
            half_warp_entry.pointer = buffer_idx;
          }

          // swap new and old entry
          u64 old_entry_int_repr = atomicCAS(&data[slot + half_lane_idx].int_repr, thread_table_entry.int_repr, half_warp_entry.int_repr);

          // make sure retrieved entry matches what was expected, so we know that CAS succeeded
          if(old_entry_int_repr != thread_table_entry.int_repr) {
            // set retry indicator
            retry = true;

            // clear buffer, if it was requested
            if(buffer_idx != GPLDA_HASH_NULL_POINTER) {
              ring_buffer[buffer_idx] = entry(false, GPLDA_HASH_NULL_POINTER, GPLDA_HASH_EMPTY, 0);
              ring_buffer_push(buffer_idx);
              half_warp_entry.pointer = GPLDA_HASH_NULL_POINTER;
            } else {
              half_warp_entry = entry(false, GPLDA_HASH_NULL_POINTER, GPLDA_HASH_EMPTY, 0);
            }
          }
        }

        // ensure entire halfwarp knows whether write succeeded
        half_warp_entry.int_repr = __shfl(half_warp_entry.int_repr, 0, warpSize/2);

        // ensure retry, if necessary, is performed on entire half warp
        retry = __ballot(retry) & half_lane_mask;
      } while(retry != 0);

      // if half warp successfully performed a write, exit the loop
      if(half_warp_entry.int_repr == entry(false, GPLDA_HASH_NULL_POINTER, GPLDA_HASH_EMPTY, 0).int_repr) {
        break;
      }
    }




    // check to make sure insertion succeeded: if it failed, return
    if(half_warp_entry.int_repr != entry(false, GPLDA_HASH_NULL_POINTER, GPLDA_HASH_EMPTY, 0).int_repr) {
      return false;
    }




    // resolve queue
    u32 finished;
    do {
      // find element to be resolved
      thread_table_entry = data[slot + half_lane_idx];
      finished = false;

      u32 half_warp_relocation = __ballot(thread_table_entry.relocate != 0) & half_lane_mask;
      u32 half_warp_pointer = __ballot(thread_table_entry.pointer != GPLDA_HASH_NULL_POINTER) & half_lane_mask;
      if(half_warp_relocation != 0) {
        // resolve relocation bit: first, broadcast pointer to entire half warp, then retrieve entry
        u32 lane_link_entry_idx = __ffs(half_warp_relocation) - 1;
        u32 half_warp_link_entry_pointer = __shfl(thread_table_entry.pointer, lane_link_entry_idx % (warpSize/2), warpSize/2);
        HashMapEntry half_warp_link_entry = ring_buffer[half_warp_link_entry_pointer];

        // figure out whether linked element should take thread's slot, or whether thread's slot needs to be moved
        if(half_warp_link_entry.relocate == 1) {
          // first linked element has a relocation bit: move it
          if(lane_idx == lane_link_entry_idx) {
            // no need to check for success: whether we succeed or fail, try again and keep going
            atomicCAS(&data[slot + half_lane_idx].int_repr, thread_table_entry.int_repr, half_warp_link_entry.int_repr);
          }
        } else {
          // element has relocation bit, but its first linked element doesn't: find slot relocated element is supposed to go in
          HashMapEntry half_warp_table_entry;
          half_warp_table_entry.int_repr = __shfl(thread_table_entry.int_repr, lane_link_entry_idx % (warpSize/2), warpSize/2);

          // find slot relocated element is supposed to go into
          i32 insert_stride = hash_slot(half_warp_table_entry.key,c,d);
          for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
            i32 insert_slot = (slot + i * insert_stride) % size;
            HashMapEntry thread_table_insert_entry = data[insert_slot + half_lane_idx];

            // check first if slot contains an empty element: if so, insert the element there - no need to check pointers because they must be null
            u32 slot_empty = __ballot(thread_table_insert_entry.key == GPLDA_HASH_EMPTY) & half_lane_mask;
            if(slot_empty != 0) {
              i32 slot_empty_lane_idx = __ffs(slot_empty) - 1;
              if(lane_idx == slot_empty_lane_idx) {
                u64 old_entry_int_repr = atomicCAS(&data[insert_slot + half_lane_idx].int_repr, thread_table_insert_entry.int_repr, thread_table_entry.int_repr);
                if(old_entry_int_repr == entry(false, GPLDA_HASH_NULL_POINTER, GPLDA_HASH_EMPTY, 0).int_repr) {
                  slot = insert_slot;
                }
              }
              // ensure entire half warp knows the new slot value, if it changed
              slot = __shfl(slot, slot_empty_lane_idx % (warpSize/2), warpSize/2);
              break;
            }

            // assuming slot is full, check pointers to see if element is there
            u32 found;
            u32 pointer;
            HashMapEntry* address = &data[insert_slot + half_lane_idx];
            do {
              // if element is found, set relocation bit on its first link
              found = false;
              if(thread_table_insert_entry.int_repr == half_warp_table_entry.int_repr) {
                found = true;
                HashMapEntry half_warp_link_entry_with_relocate = half_warp_link_entry;
                half_warp_link_entry_with_relocate.relocate = 1;
                // no need to check for success: whether we succeed or fail, try again and keep going
                atomicCAS(&ring_buffer[half_warp_link_entry_pointer].int_repr, half_warp_link_entry.int_repr, half_warp_link_entry_with_relocate.int_repr);
              }
              found = __ballot(found == true) & half_lane_mask;

              // if pointers are present, follow them and check again
              pointer = false;
              if(found == 0 && thread_table_insert_entry.pointer != GPLDA_HASH_NULL_POINTER) {
                pointer = true;
                address = &ring_buffer[thread_table_insert_entry.pointer];
                thread_table_insert_entry = *address;
              }
              pointer = __ballot(pointer == true) & half_lane_mask;
            } while(found == 0 && pointer != 0);

            // exit if we found an element
            if(found != 0) {
              break;
            }

            // after pointers have been exhausted, check if element should be evicted, and insert into queue
            u32 evict = __ballot(key_distance(thread_table_insert_entry.key, insert_slot) < i) & half_lane_mask;
            if(evict != 0 && lane_idx == __ffs(evict) - 1) {
              // grab slot from ring buffer
              u32 buffer_idx = ring_buffer_pop();
              ring_buffer[buffer_idx] = half_warp_table_entry;

              // prepare entry for insertion
              HashMapEntry thread_table_insert_entry_with_pointer = thread_table_insert_entry;
              thread_table_insert_entry_with_pointer.pointer = buffer_idx;

              // insert entry, returning value to ring buffer if insert failed
              u64 old_entry_int_repr = atomicCAS(&address->int_repr, thread_table_insert_entry.int_repr, thread_table_insert_entry_with_pointer.int_repr);
              if(old_entry_int_repr != thread_table_insert_entry.int_repr) {
                ring_buffer_push(buffer_idx);
              }
            }

            // exit if we evicted
            if(evict != 0) {
              break;
            }

            // if we're at the last iteration and haven't exited the loop yet, return indicating failure
            if(i == GPLDA_HASH_MAX_NUM_LINES - 1) {
              return false;
            }
          }
        }
      } else if(half_warp_pointer != 0){
        // we have pointers, but no relocation bit: resolve pointer on first thread that found it
        if(lane_idx == __ffs(half_warp_pointer) - 1) {
          // set relocation bit
          HashMapEntry thread_new_entry = thread_table_entry;
          thread_new_entry.relocate = 1;

          // no need to check for success: whether we succeed or fail, try again and keep going
          atomicCAS(&data[slot + half_lane_idx].int_repr, thread_table_entry.int_repr, thread_new_entry.int_repr);
        }
      } else {
        // no relocation bit or pointer present, so we must have either inserted to an empty slot or accumulated existing element
        finished = true;
      }

      // ensure entire half warp finishes
      finished = __ballot(finished) & half_lane_mask;
    } while(finished == 0);

    // return empty element indicating success, or element that exceeded its maximum number of lines indicating failure
    return true;
  }

  __device__ __forceinline__ void accumulate2(u32 key, i32 diff) {
    // try to accumulate
    i32 success = try_accumulate2(key, diff);

    if(!success) {
      printf("accumulate2 failed");
    }

    // rebuild if too large
//    sync();
//    if(needs_rebuild == 1) {
//      rebuild();
//    }
  }
};

}
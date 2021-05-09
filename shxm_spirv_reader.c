#include "shxm.h"
#include "shxm_private.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

shxm_spirv_ent_t* 
shxm_private_read_spirv(uint32_t* ir, int len, int* out_count){
    shxm_spirv_ent_t* ent;
    uint32_t magic;
    uint32_t version;
    uint32_t bound;
    uint32_t zero;
    uint32_t oplen;
    uint32_t op;
    int i;
    if(len < 5){
        printf("ERROR: Too short(%d)\n", len);
        return 0;
    }

    magic = ir[0];
    version = ir[1];
    bound = ir[3];
    zero = ir[4];

    printf("Reading: magic = %x version = %x bound = %d zero = %d\n",
           magic, version, bound, zero);

    ent = malloc(sizeof(shxm_spirv_ent_t)*bound);
    if(ent){
        memset(ent, 0, sizeof(shxm_spirv_ent_t*)*bound);
        i = 5;
        while(i<len){
            oplen = (ir[i]) >> 16;
            op = (ir[i]) & 0xffff;
            printf("[%d] op = %d len = %d(%d)\n", i, op, oplen,len);
            if(oplen == 0){
                break;
            }
            i+=oplen;
        }
    }
    if(out_count){
        *out_count = bound;
    }
    return ent;
}


#include <stdlib.h>
#include <stdio.h>
#include "shxm.h"
#include "shxm_private.h"

SHXM_API int
shxm_program_link(shxm_ctx_t* ctx, shxm_program_t* prog){
    shxm_spirv_intr_t* vintr;
    shxm_spirv_intr_t* fintr;

    if(!prog->vertex_shader){
        printf("ERROR: No vertex shader.\n");
        return 1;
    }
    if(!prog->fragment_shader){
        printf("ERROR: No fragment shader.\n");
        return 1;
    }
    if(!prog->vertex_shader->ir){
        printf("ERROR: Vertex shader not compiled.\n");
        return 1;
    }
    if(!prog->fragment_shader->ir){
        printf("ERROR: Fragment shader not compiled.\n");
        return 1;
    }

    vintr = shxm_private_read_spirv(prog->vertex_shader->ir,
                                    prog->vertex_shader->ir_len);
    fintr = shxm_private_read_spirv(prog->fragment_shader->ir,
                                    prog->fragment_shader->ir_len);

    // FIXME: Implement this.
    return 1;
}


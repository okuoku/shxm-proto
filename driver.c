#include <stdio.h>
#include <stdlib.h>

#include "shxm.h"
#include "shxm_private.h"

static void*
readfile(const char* fn, size_t* out_size){
    FILE* fp;
    size_t sz;
    char* r;
    fp = fopen(fn, "rb");
    fseek(fp, 0, SEEK_END);
    sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    r = malloc(sz+1);
    r[sz] = 0;
    fread(r, sz, 1, fp);
    *out_size = sz;
    return r;
}

int
main(int ac, char** av){
    char* src;
    size_t len;
    shxm_ctx_t* ctx;
    shxm_shader_t* shf;
    shxm_shader_t* shv;
    shxm_program_t* prog;

    ctx = shxm_init();
    shf = shxm_shader_create(ctx, SHXM_SHADER_STAGE_FRAGMENT);
    shv = shxm_shader_create(ctx, SHXM_SHADER_STAGE_VERTEX);
    src = readfile(SOURCEPATH "/simple.frag", &len);
    shxm_shader_source(ctx, shf, src, len);
    src = readfile(SOURCEPATH "/simple.vert", &len);
    shxm_shader_source(ctx, shv, src, len);

    printf("== FRAG ==\n");
    shxm_shader_compile(ctx, shf);
    printf("== VERT ==\n");
    shxm_shader_compile(ctx, shv);

    prog = shxm_program_create(ctx);
    shxm_program_attach(ctx, prog, shf);
    shxm_program_attach(ctx, prog, shv);

    printf("== PreLink ==\n");
    printf("Fsize = %d, Vsize = %d\n", 
           shf->ir_len,
           shv->ir_len);
    shxm_program_link(ctx, prog);

    return 0;
}

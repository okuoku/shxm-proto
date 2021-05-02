#include <stdio.h>
#include <stdlib.h>

void shxm_init(void);
void shxm_deinit(void);
void* shxm_build(int mode, const char* source);

static void*
readfile(const char* fn){
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
    return r;
}

int
main(int ac, char** av){
    char* src;
    shxm_init();
    src = readfile(SOURCEPATH "/phys.frag");
    shxm_build(0 /* frag */, src);
    shxm_deinit();
}

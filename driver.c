#include <stdio.h>
#include <stdlib.h>

void shxm_init(void);
void shxm_deinit(void);

int
main(int ac, char** av){
    shxm_init();
    shxm_deinit();
}

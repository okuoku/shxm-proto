#include "ShaderLang.h"

extern "C" void
shxm_init(void){
    (void)ShInitialize();
}

extern "C" void
shxm_deinit(void){
    (void)ShFinalize();
}

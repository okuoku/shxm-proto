#include "ShaderLang.h"
#include "SPIRV/GlslangToSpv.h"
#include "SPIRV/disassemble.h"
#include <iostream>

namespace glslang {
    extern TBuiltInResource DefaultTBuiltInResource;
};

extern "C" void
shxm_init(void){
    (void)glslang::InitializeProcess();
}

extern "C" void
shxm_deinit(void){
    (void)glslang::FinalizeProcess();
}

extern "C" void*
shxm_build(int mode, const char* source){
    glslang::TShader::ForbidIncluder includer;
    glslang::TShader* ts;
    switch(mode){
        case 0:
            ts = new glslang::TShader(EShLangFragment);
            break;
        case 1:
            ts = new glslang::TShader(EShLangVertex);
            break;
        default:
            return 0;
    }

    ts->setStrings(&source, 1);
    ts->setAutoMapBindings(true);
    ts->setAutoMapLocations(true);
    // ts->setInvertY(true);
    ts->setEnvInput(glslang::EShSourceGlsl,
                    (mode == 0)?EShLangFragment : EShLangVertex,
                    glslang::EShClientOpenGL,
                    100);
    /*
    ts->setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    ts->setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
    */

    if(ts->parse(&glslang::DefaultTBuiltInResource,
                 100,
                 false,
                 EShMsgDefault,
                 includer)){
        std::vector<unsigned int> spirv;
        spv::SpvBuildLogger logger;
        glslang::SpvOptions opt;

        printf("Success.\n%s\n%s\n",
               ts->getInfoLog(),
               ts->getInfoDebugLog());
        glslang::GlslangToSpv(*ts->getIntermediate(),
                              spirv,
                              &logger,
                              &opt);
        printf("%s\n", logger.getAllMessages().c_str());
        spv::Disassemble(std::cout, spirv);
    }else{
        printf("Fail.\n%s\n%s\n",
               ts->getInfoLog(),
               ts->getInfoDebugLog());
    }



    return static_cast<void*>(ts);
}

extern "C" void
shxm_release(void* obj){
}

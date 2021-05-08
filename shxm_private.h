#ifndef __YUNI_SHXM_PRIVATE_H
#define __YUNI_SHXM_PRIVATE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// }

struct shxm_ctx_s {
    int dummy;
};

struct shxm_slot_s {
    cwgl_var_type_t type;
    const char* name;
};

typedef struct shxm_slot_s shxm_slot_t;

struct shxm_uniform_s {
    shxm_slot_t* slot;
    unsigned int offset;
    unsigned int size;
};

typedef struct shxm_uniform_s shxm_uniform_t;

struct shxm_attribute_s {
    shxm_slot_t* slot;
};

typedef struct shxm_attribute_s shxm_attribute_t;

struct shxm_varying_s {
    shxm_slot_t* slot;
};

typedef struct shxm_varying_s shxm_varying_t;

struct shxm_shader_s {
    int refcnt;
    shxm_shader_stage_t type;
    char* source; /* GLSL Source */
    unsigned int source_len;
    uint32_t* ir; /* SPIR-V IR (Unpatched) */
    unsigned int ir_len;
};
typedef struct shxm_shader_s shxm_shader_t;

#define SHXM_MAX_SLOTS 512
#define SHXM_MAX_ATTRIBUTES 256
#define SHXM_MAX_VARYINGS 256
#define SHXM_MAX_UNIFORMS 256
struct shxm_program_s {
    /* Pre-link fields */
    shxm_shader_t* vertex_shader;
    shxm_shader_t* fragment_shader;

    /* Post-link fields */
    shxm_slot_t slot[SHXM_MAX_SLOTS];
    unsigned int slot_count;
    shxm_uniform_t uniform[SHXM_MAX_UNIFORMS];
    unsigned int uniform_count;
    shxm_attribute_t attribute[SHXM_MAX_ATTRIBUTES];
    unsigned int attribute_count;
    shxm_varying_t varying[SHXM_MAX_VARYINGS];
    unsigned int varying_count;

    uint32_t* vertex_ir;
    unsigned int vertex_ir_len;
    uint32_t* fragment_ir;
    unsigned int fragment_ir_len;
};


// {
#ifdef __cplusplus
};
#endif

#endif


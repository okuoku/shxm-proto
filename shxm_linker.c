#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "shxm.h"
#include "shxm_private.h"

static int /* zero on success */
resolve_type(shxm_spirv_ent_t* ent, int32_t* ir, int id){
    int r;
    int op;
    int len;
    int offs;
    int tgt;
    int dim;
    if(ent[id].type == CWGL_VAR_UNKNOWN){
        op = ent[id].op;
        offs = ent[id].offs;
        switch(op){
            case 19: /* OpTypeVoid */
                ent[id].type = CWGL_VAR_DUMMY;
                break;
            case 21: /* OpTypeInt */
                ent[id].type = CWGL_VAR_INT;
                ent[id].array_length = 0;
                ent[id].width = ir[offs+2];
                ent[id].is_signed = ir[offs+3];
                break;
            case 22: /* OpTypeFloat */
                ent[id].type = CWGL_VAR_FLOAT;
                ent[id].array_length = 0;
                ent[id].width = ir[offs+2];
                ent[id].is_signed = 0;
                break;
            case 23: /* OpTypeVector */
                tgt = ir[offs+2];
                len = ir[offs+3];
                ent[id].array_length = 0;
                ent[id].width = 0;
                ent[id].is_signed = 0;
                r = resolve_type(ent, ir, tgt);
                if(r){
                    return r;
                }
                switch(ent[tgt].type){
                    // FIXME: How to handle BOOL??
                    case CWGL_VAR_FLOAT:
                        switch(len){
                            case 2:
                                ent[id].type = CWGL_VAR_FLOAT_VEC2;
                                break;
                            case 3:
                                ent[id].type = CWGL_VAR_FLOAT_VEC3;
                                break;
                            case 4:
                                ent[id].type = CWGL_VAR_FLOAT_VEC4;
                                break;
                            default:
                                printf("ERROR: Invalid vector width(%d)",
                                       len);
                                return 1;
                        }
                        break;
                    case CWGL_VAR_INT:
                        switch(len){
                            case 2:
                                ent[id].type = CWGL_VAR_INT_VEC2;
                                break;
                            case 3:
                                ent[id].type = CWGL_VAR_INT_VEC3;
                                break;
                            case 4:
                                ent[id].type = CWGL_VAR_INT_VEC4;
                                break;
                            default:
                                printf("ERROR: Invalid vector width(%d)",
                                       len);
                                return 1;
                        }
                        break;
                    default:
                        printf("ERROR: Invalid vector CWGL type(%d)\n", 
                               (int)ent[tgt].type);
                        return 1;
                }
                break;
            case 24: /* OpTypeMatrix */
                tgt = ir[offs+2];
                len = ir[offs+3];
                ent[id].array_length = 0;
                ent[id].width = 0;
                ent[id].is_signed = 0;
                switch(len){
                    case 2:
                        if(ent[tgt].type != CWGL_VAR_FLOAT_VEC2){
                            printf("ERROR: Invalid matrix col type(%d)\n",
                                   (int)ent[tgt].type);
                        }
                        ent[id].type = CWGL_VAR_FLOAT_MAT2;
                        break;
                    case 3:
                        if(ent[tgt].type != CWGL_VAR_FLOAT_VEC3){
                            printf("ERROR: Invalid matrix col type(%d)\n",
                                   (int)ent[tgt].type);
                        }
                        ent[id].type = CWGL_VAR_FLOAT_MAT3;
                        break;
                    case 4:
                        if(ent[tgt].type != CWGL_VAR_FLOAT_VEC4){
                            printf("ERROR: Invalid matrix col type(%d)\n",
                                   (int)ent[tgt].type);
                        }
                        ent[id].type = CWGL_VAR_FLOAT_MAT4;
                        break;
                    default:
                        printf("ERROR: Invalid matrix col count(%d)\n",
                               len);
                        return 1;
                }
                break;
            case 25: /* OpTypeImage */
                dim = ir[offs+3];
                ent[id].array_length = 0;
                ent[id].width = 0;
                ent[id].is_signed = 0;
                switch(dim){
                    case 1: /* 2D */
                        ent[id].type = CWGL_VAR_SAMPLER_2D;
                        break;
                    case 3: /* Cube */
                        ent[id].type = CWGL_VAR_SAMPLER_CUBE;
                        break;
                    case 0: /* 1D */
                    case 2: /* 3D */
                    case 4: /* Rect */
                    case 5: /* Buffer */
                    case 6: /* SubpassData */
                    default:
                        printf("ERROR: Unrecognised image dimension(%d)\n",
                               dim);
                        return 1;
                }
                break;
            case 26: /* OpTypeSampler */
                printf("ERROR: OpTypeSampler(FIXME)\n");
                return 1;
            case 27: /* OpTypeSampledImage */
                tgt = ir[offs+2];
                ent[id].array_length = 0;
                ent[id].width = 0;
                ent[id].is_signed = 0;
                r = resolve_type(ent, ir, tgt);
                if(r){
                    return r;
                }
                ent[id].type = ent[tgt].type;
                break;
            case 32: /* OpTypePointer */
                tgt = ir[offs+3];
                r = resolve_type(ent, ir, tgt);
                if(r){
                    return r;
                }
                ent[id].array_length = ent[tgt].array_length;
                ent[id].width = ent[tgt].width;
                ent[id].is_signed = ent[tgt].is_signed;
                ent[id].type = ent[tgt].type;
                break;
            case 33: /* OpTypeFunction */
                printf("ERROR: Tried to resolve function type op(%d)\n", op);
                return 1;
            default:
                printf("ERROR: Tried to resolve unknown type op(%d)\n", op);
                return 1;
        }
    }
    return 0;
}

static int
fill_slots(shxm_program_t* prog, shxm_spirv_intr_t* intr, int phase){
    /* Look for OpVariable Input/Output/Uniform/UniformConstant */
    int id;
    int failed = 0;
    enum {
        UNKNOWN,
        INPUT,
        OUTPUT,
        UNIFORM,
        UNIFORM_CONSTANT
    } varusage;
    char* varname;
    int varclass;
    int varwidth;
    int r;
    int v;
    int typeid;
    shxm_slot_t* varslot;
    cwgl_var_type_t vartype;
    int32_t* ir = (phase == 0) ? prog->vertex_shader->ir : 
        prog->fragment_shader->ir;
    if(phase == 0){
        /* Reset current slot data */
        prog->slot_count = 0;
        prog->uniform_count = 0;
        prog->attribute_count = 0;
        prog->varying_count = 0;
    }
    /* Pass1: Pickup variable decl. */
    for(id=0;id!=intr->ent_count;id++){
        if(intr->ent[id].op == 59 /* OpVariable */){
            if(intr->ent[id].name){
                varname = (char*)&ir[intr->ent[id].name+2];
            }else{
                varname = NULL;
            }
            varusage = UNKNOWN;
            varclass = ir[intr->ent[id].offs+3];
            switch(varclass){
                case 0: /* UniformConstant */
                    varusage = UNIFORM_CONSTANT;
                    break;
                case 1: /* Input */
                    varusage = INPUT;
                    break;
                case 2: /* Uniform */
                    printf("ERROR: Uniform??\n");
                    varusage = UNIFORM;
                    break;
                case 3: /* Output */
                    varusage = OUTPUT;
                    break;
                default:
                    /* Unknown */
                    break;
            }

            if(phase == 0){
                varslot = &prog->slot[prog->slot_count];
                memset(varslot, 0, sizeof(shxm_slot_t));
                prog->slot_count++;
            }else{
                varslot = NULL;
                /* First, search existing variable with matching name */
                for(v=0;v!=prog->slot_count;v++){
                    if(!strncmp(varname, prog->slot[v].name, 255)){
                        varslot = &prog->slot[v];
                        break;
                    }
                }
                if(! varslot){
                    varslot = &prog->slot[prog->slot_count];
                    memset(varslot, 0, sizeof(shxm_slot_t));
                    prog->slot_count++;
                }
            }
            if(varslot){
                typeid = ir[intr->ent[id].offs+1];
                varslot->id[phase] = id;
                varslot->name = varname;
                r = resolve_type(intr->ent, ir, typeid);
                if(r){
                    return r;
                }
                varslot->type = intr->ent[typeid].type;
                printf("var:%d:%s:%d (type %d)\n",phase, varname, (int)varusage,
                       varslot->type);
            }else{
                printf("(ignored) var:%d:%s:%d\n",phase, varname, (int)varusage);
            }
        }
    }
    /* Pass2: Resolve type and decoration chain */
    for(v=0;v!=prog->slot_count;v++){
    }
    return failed;
}

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

    if(fill_slots(prog, vintr, 0)){
        printf("ERROR: Failed to fill vertex shader variable slots.\n");
        // FIXME: Release vintr, fintr
        return 1;
    }
    if(fill_slots(prog, fintr, 1)){
        printf("ERROR: Failed to fill fragment shader variable slots.\n");
        // FIXME: Release vintr, fintr
        return 1;
    }

    // FIXME: Implement this.
    return 1;
}


#ifndef _USER_CPU_COMMON_H_
#define _USER_CPU_COMMON_H_

/*
 * Note:
 *
 * If user revises the parameter struct, the compiled bmodel that contains your revised
 * cpu layer parameter will be not supported. Please recompile the bmodel that contains
 * the revised cpu layer parameter.
 *
 * If user revises the process program and does not revise the cpu parameter, the
 * compiled bmodel can be still supported. User does not need to recompile the bmoel.
 *
 */

typedef enum {
    USER_EXP = 0,
    USER_CPU_UNKNOW
} USER_CPU_LAYER_TYPE_T;


typedef struct user_cpu_exp_param
{
    float inner_scale_;
    float outer_scale_;
} user_cpu_exp_param_t;


typedef struct user_cpu_param
{
    int op_type;   /* USER_CPU_LAYER_TYPE_T */
    union {
        user_cpu_exp_param_t exp;
        /* notice: please add other cpu layer param here */
    }u;
} user_cpu_param_t;



#endif /* _USER_CPU_COMMON_H_ */

register_custom_op(absadd)
register_custom_ppl_op(addconst)
register_custom_op(ceiladd)
register_custom_op(swapchannel)
register_custom_op(crop)
register_custom_op(preprocess)
# scaleadd: multi-input (A,B) -> multi-output (A*scale, A*scale+B)
register_custom_ppl_op(scaleadd)    # via .pl kernel

register_custom_local_op(absadd)
register_custom_ppl_local_op(addconst)
register_custom_local_op(ceiladd)

register_custom_local_bfsz(absadd)
register_custom_local_bfsz(addconst)
register_custom_local_bfsz(ceiladd)
register_custom_local_bfsz(preprocess)

const char* jsi_GetDirective(Jsi_Interp *interp, Jsi_OpCodes *ops, const char *str) {
    if (!ops) return NULL;
    if (!ops->code_len) return NULL;
    if (ops->codes[0].op != OP_PUSHSTR || !ops->codes[0].data) return NULL;
    if (Jsi_Strncmp((char*)ops->codes[0].data, str, Jsi_Strlen(str))) return NULL;
    return (char*)ops->codes[0].data;
}
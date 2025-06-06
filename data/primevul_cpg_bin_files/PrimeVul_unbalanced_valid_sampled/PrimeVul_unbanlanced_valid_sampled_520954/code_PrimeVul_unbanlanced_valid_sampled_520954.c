static Jsi_RC CDataStructGetCmd(Jsi_Interp *interp, Jsi_Value *args, Jsi_Value *_this,
                                Jsi_Value **ret, Jsi_Func *funcPtr)
{
    
    char *arg1 = Jsi_ValueArrayIndexToStr(interp, args, 0, NULL);
    Jsi_StructSpec *sl = jsi_csStructGet(interp, arg1);

    if (!sl)
        return Jsi_LogError("unkown struct: %s", arg1);
    Jsi_DString dStr = {};
    Jsi_RC rc = CDataStructGetDfn(interp, sl, &dStr);
    if (JSI_OK == rc)
        rc = Jsi_JSONParse(interp, Jsi_DSValue(&dStr), ret, 0);
    Jsi_DSFree(&dStr);
    return rc;
}
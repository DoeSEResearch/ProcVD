static Jsi_RC CDataStructSchemaCmd(Jsi_Interp *interp, Jsi_Value *args, Jsi_Value *_this,
    Jsi_Value **ret, Jsi_Func *funcPtr) {
    char *arg1 = Jsi_ValueArrayIndexToStr(interp, args, 0, NULL);
    Jsi_StructSpec *sf = jsi_csStructFields(interp, arg1);

    if (!sf)
        return Jsi_LogError("unkown struct: %s", arg1);;

    Jsi_DString dStr = {};
    Jsi_OptionsData(interp, (Jsi_OptionSpec*)sf, &dStr, 1);
    Jsi_ValueFromDS(interp, &dStr, ret);
    Jsi_DSFree(&dStr);
    return JSI_OK;
}
static Jsi_RC jsi_NumberIsFiniteCmd(Jsi_Interp *interp, Jsi_Value *args, Jsi_Value *_this,
    Jsi_Value **ret, Jsi_Func *funcPtr, int op)
{
    Jsi_Number num;
    Jsi_Value *v;
    bool b = 0;
    int skip = 0;
    ChkStringN(_this, funcPtr, v);
    if (Jsi_GetNumberFromValue(interp, v, &num) != JSI_OK)
        return JSI_ERROR;
    switch (op) {
        case 1: b = Jsi_NumberIsFinite(num); break;
        case 2: b = Jsi_NumberIsInteger(num); break;
        case 3: b = Jsi_NumberIsNaN(num); break;
        case 4: b = Jsi_NumberIsSafeInteger(num); break;
    }
    Jsi_ValueMakeBool(interp, ret, b);
    skip++;
    return JSI_OK;
}
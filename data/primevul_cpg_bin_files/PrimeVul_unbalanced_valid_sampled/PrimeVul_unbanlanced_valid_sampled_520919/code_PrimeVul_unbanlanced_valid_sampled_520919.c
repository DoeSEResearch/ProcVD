static Jsi_RC CDataSetCmd_(Jsi_Interp *interp, Jsi_Value *args, Jsi_Value *_this,
                             Jsi_Value **ret, Jsi_Func *funcPtr, int op)
{
    UdcGet(cd, _this, funcPtr);
    uchar *dptr = NULL;
    Jsi_Value *arg = Jsi_ValueArrayIndex(interp, args, 0);
    char kbuf[BUFSIZ];
    void *key = kbuf;
    bool isNull = Jsi_ValueIsNull(interp, arg);
    if (isNull) {
        if (cd->mapPtr || cd->arrSize)
            return Jsi_LogError("null key used with c-array/map");
    } else {
        if (!cd->mapPtr && cd->arrSize<=0)
            return Jsi_LogError("must be array/map");
        if (JSI_OK != jsi_csGetKey(interp, cd, arg, &key, sizeof(kbuf), 1))
            return JSI_ERROR;
    }

    dptr = (uchar*)cd->data;
    if (isNull) {
    } else if (cd->mapPtr) {
        Jsi_MapEntry *mPtr = Jsi_MapEntryFind(*cd->mapPtr, key);
        if (mPtr)
            dptr = (uchar*)Jsi_MapValueGet(mPtr);
        else {
            bool isNew;
            if (cd->maxSize && Jsi_MapSize(*cd->mapPtr)>=cd->maxSize)
                return Jsi_LogError("map would exceeded maxSize: %d", cd->maxSize);
            if (!cd->noAuto)
                mPtr = Jsi_MapEntryNew(*cd->mapPtr, key, &isNew);
            if (!mPtr)
                return Jsi_LogError("arg 1: key not found [%s]", Jsi_ValueToString(interp, arg, NULL));
            Jsi_StructSpec *sl = cd->sl;
            dptr = (uchar*)Jsi_Calloc(1, sl->size);
            Jsi_MapValueSet(mPtr, dptr);
            jsi_csStructInit(sl, dptr);
        }
    } else if (!cd->arrSize)
        return Jsi_LogError("expected a c-array or map");
    else {
        uint kind = (uintptr_t)key;
        if (kind>=cd->arrSize)
            return Jsi_LogError("array index out of bounds: %d not in 0,%d", kind, cd->arrSize-1);
        dptr = ((uchar*)cd->data) + cd->sl->size*kind;
        if (cd->isPtrs)
            dptr = ((uchar*)cd->data) + sizeof(void*)*kind;
        else if (cd->isPtr2)
            dptr = (uchar*)(*(void**)dptr) + sizeof(void*)*kind;
    }
    int argc = Jsi_ValueGetLength(interp, args);
    Jsi_Value *arg2 = Jsi_ValueArrayIndex(interp, args, 1);
    if (op == 2 && argc == 2) {
    } else if (argc == 2) {
        if (!Jsi_ValueIsObjType(interp, arg2, JSI_OT_OBJECT))
            return Jsi_LogError("arg 3: last must be an object with 3 args");
        return Jsi_OptionsConf(interp, (Jsi_OptionSpec*)cd->sf, dptr, arg2, ret, 0);
    } else if (argc != 3)
        return Jsi_LogError("expected 2 or 3 args");
    const char *cp;
    if (!(cp = Jsi_ValueString(interp, arg2, NULL)))
        return Jsi_LogError("with 3 args, string expected for arg 2");
    Jsi_Value *arg3 = Jsi_ValueArrayIndex(interp, args, 2);
    if (op == 2) {
        if (arg3) {
            if (!Jsi_ValueIsNumber(interp, arg3))
                return Jsi_LogError("expected number");
        } else {
            arg3 = Jsi_ValueNewNumber(interp, 1);
            Jsi_IncrRefCount(interp, arg3);
        }
    }
    Jsi_RC rc = Jsi_OptionsSet(interp, (Jsi_OptionSpec*)cd->sf, dptr, cp, arg3, (op==2?JSI_OPTS_INCR:0));
    if (op==2 && argc==2)
        Jsi_DecrRefCount(interp, arg3);
    if (op == 2 && rc == JSI_OK) {
        if (interp->strict && Jsi_NumberIsNaN(interp->cdataIncrVal))
            return Jsi_LogError("result is NaN");
        Jsi_ValueMakeNumber(interp, ret, interp->cdataIncrVal);
    }
    return rc;
}
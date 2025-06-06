static Jsi_RC FilesysWriteCmd(Jsi_Interp *interp, Jsi_Value *args, Jsi_Value *_this,
    Jsi_Value **ret, Jsi_Func *funcPtr)
{
    int  sum = 0, n, m;
    UdfGet(udf, _this, funcPtr);
    char *buf = Jsi_ValueArrayIndexToStr(interp, args, 0, &m);

    if (!udf->filename) {
        goto bail;
    }
    while (m > 0 && sum < MAX_LOOP_COUNT && (n = Jsi_Write(interp, udf->chan, buf, m)) > 0) {
        /* TODO: limit max size. */
        sum += n;
        m -= n;
    }
bail:
    Jsi_ValueMakeNumber(interp, ret, (Jsi_Number)sum);
    return JSI_OK;
}
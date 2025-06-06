static Jsi_RC jsi_csStructInit(Jsi_StructSpec * sl, uchar * data)
{
    /* Jsi_OptionSpec *sf; */
    assert(sl);
    if (!data) {
        fprintf(stderr, "missing data at %s:%d", __FILE__, __LINE__);
        return JSI_ERROR;
    }
    if (sl->custom)
        memcpy(data, sl->custom, sl->size);
    else if (sl->type && sl->type->extra)
        memcpy(data, sl->type->extra, sl->size);
    else
        memset(data, 0, sl->size);

    if (sl->ssig)
        *(Jsi_Sig *) data = sl->ssig;
    return JSI_OK;
}
static Jsi_RC CDataStructGetDfn(Jsi_Interp *interp, Jsi_StructSpec * sl, Jsi_DString *dsPtr)
{
    
    Jsi_StructSpec *sf;
    Jsi_DString eStr = {};
    sf = jsi_csGetFieldSpec(sl->extData);
    Jsi_DSPrintf(dsPtr, "{ \"name\": \"%s\", \"size\":%d", sl->name, sl->size);
    if (sl->flags)
        Jsi_DSPrintf(dsPtr, ", \"flags\":0x%" PRIx64, sl->flags);
    if (sl->help && sl->help[0]) {
        Jsi_DSAppend(dsPtr, ", \"label\":", Jsi_JSONQuote(interp, sl->help, -1, &eStr), NULL);
        Jsi_DSFree(&eStr);
    }
    if (sl->ssig)
        Jsi_DSPrintf(dsPtr, ", \"sig\":0x%x", sl->ssig);
    Jsi_DSAppend(dsPtr, ", \"fields\":[", NULL);
#define SSIFNV(s) (s?"\"":""), (s?s:"null"), (s?"\"":"")
    while (sf->id != JSI_OPTION_END) {
        Jsi_DSPrintf(dsPtr, " { \"name\":\"%s\",  \"id\":%s%s%s, \"size\":%d, \"bitsize\":%d,"
            "\"offset\":%d, , \"bitoffs\":%d, \"isbit\":%d, \"label\":",
             sf->name, SSIFNV(sf->tname), sf->size, sf->bits,
             sf->offset, sf->boffset, sf->flags&JSI_OPT_IS_BITS?1:0 );
        if (sf->help && sf->help[0]) {
            Jsi_DSAppend(dsPtr, Jsi_JSONQuote(interp, sf->help, -1, &eStr), NULL);
            Jsi_DSFree(&eStr);
        } else
            Jsi_DSAppend(dsPtr,"\"\"", NULL);
        Jsi_DSAppend(dsPtr, "}", NULL);
        sf++;
    }
    Jsi_DSAppend(dsPtr, "]}", NULL);
    return JSI_OK;
}
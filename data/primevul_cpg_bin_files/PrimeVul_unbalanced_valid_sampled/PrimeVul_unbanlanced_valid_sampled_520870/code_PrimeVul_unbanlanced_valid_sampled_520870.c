static Jsi_RC jsi_DoneCData(Jsi_Interp *interp)
{
    if (!interp->SigHash) return JSI_OK;
    Jsi_HashDelete(interp->SigHash);
    Jsi_HashDelete(interp->StructHash);
    Jsi_HashDelete(interp->EnumHash);
    Jsi_HashDelete(interp->EnumItemHash);
    Jsi_HashDelete(interp->TYPEHash);
    Jsi_HashDelete(interp->CTypeHash);
    return JSI_OK;
}
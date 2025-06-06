void jsi_PstatePush(jsi_Pstate *ps)
{
    Jsi_Interp *interp = ps->interp;
    if (interp->cur_scope >= (int)(JSI_MAX_SCOPE - 1)) {
        Jsi_LogBug("Scope chain too short");
        return;
    }
    interp->cur_scope++;
}
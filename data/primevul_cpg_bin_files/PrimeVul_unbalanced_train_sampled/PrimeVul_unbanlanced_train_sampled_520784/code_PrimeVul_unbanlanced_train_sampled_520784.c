static jsi_CaseList *caselist_insert(jsi_Pstate *pstate, jsi_CaseList *a, jsi_CaseExprStat *es)
{
    jsi_CaseList *b = (jsi_CaseList*)Jsi_Calloc(1,sizeof(*b));
    a->sig = JSI_SIG_CASELIST;
    b->es = es;
    a->tail->next = b;
    a->tail = b;
    return a;
}